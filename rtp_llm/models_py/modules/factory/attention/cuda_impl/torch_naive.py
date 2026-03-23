"""Torch Naive Attention Backend - Fallback implementation using PyTorch's scaled_dot_product_attention.

This module provides a fallback attention implementation that uses PyTorch's native
scaled_dot_product_attention function. It serves as a lowest-priority backend that
works in any environment, useful for debugging and development.

Reference: SGLang's torch_native_backend.py
"""

import logging
from typing import Optional

import torch
from flash_kmeans import batch_kmeans_Euclid
from flash_kmeans.incremental_kmeans import IncrementalKMeans
from torch.nn.functional import scaled_dot_product_attention

from rtp_llm.models_py.modules.factory.attention import common
from rtp_llm.models_py.modules.factory.attention.cuda_impl.kv_cache_write_op import (
    KVCacheWriteOp,
)
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.ops import AttentionConfigs, ParallelismConfig
from rtp_llm.ops.compute_ops import (
    FusedRopeKVCacheDecodeOp,
    FusedRopeKVCachePrefillOpQKVOut,
    KVCache,
    PyAttentionInputs,
    rtp_llm_ops,
)

# ============================================================================
# Dummy FMHA Params for Interface Compatibility
# ============================================================================


class DummyFMHAParams:
    """Dummy FMHA params for TorchNaive implementation.

    This class provides interface compatibility with PyModelOutputs which expects
    an fmha_params object. Since TorchNaive doesn't use FlashInfer's FMHA operations,
    we provide a minimal dummy implementation.
    """

    def fill_params(
        self,
        sequence_lengths,
        input_lengths,
        kv_cache_block_id_host,
        batch_size,
        seq_size_per_block,
    ):
        """Dummy implementation for CUDA graph compatibility.

        This method is required by the ParamsBase interface but is not used
        by TorchNaive since it doesn't participate in CUDA graph execution.
        """
        pass


# ============================================================================
# K-Clustering Utilities for Attention Acceleration
# ============================================================================

# 全局聚类信息缓存
_CLUSTER_CACHE = {}  # key: "layer_{id}_seq_{idx}_head_{idx}" -> cluster_info


def _kmeans_clustering(
    k: torch.Tensor,  # [seq_len, head_dim]
    num_clusters: int,
    max_iters: int = 10,
) -> tuple[torch.Tensor, torch.Tensor, list]:
    """K-Means clustering using flash_kmeans.

    Args:
        k: Key tensor for clustering [seq_len, head_dim]
        num_clusters: Number of clusters
        max_iters: Maximum iterations

    Returns:
        centroids: [num_clusters, head_dim] cluster centers
        labels: [seq_len] cluster assignment per token
        cluster_indices: list[list[int]] tokens grouped by cluster
    """
    # Add batch dimension: [seq_len, head_dim] -> [1, seq_len, head_dim]
    k_batched = k.unsqueeze(0)

    # Call flash_kmeans
    cluster_ids, centroids, n_iters = batch_kmeans_Euclid(
        k_batched,
        num_clusters,
        max_iters=max_iters,
        tol=1e-4,
        init_centroids=None,
        verbose=False,
    )

    # Remove batch dimension
    labels = cluster_ids.squeeze(0)  # [seq_len]
    centroids = centroids.squeeze(0)  # [num_clusters, head_dim]

    # Build cluster_indices (list of lists)
    cluster_indices = [[] for _ in range(num_clusters)]
    for token_idx, cluster_id in enumerate(labels.tolist()):
        cluster_indices[cluster_id].append(token_idx)

    return centroids, labels, cluster_indices


def _assign_to_cluster(
    k_new: torch.Tensor,  # [head_dim] 单个新 K
    centroids: torch.Tensor,  # [num_clusters, head_dim]
) -> int:
    """分配新 K 到最近的簇，返回簇索引."""
    distances = torch.norm(centroids - k_new.unsqueeze(0), dim=1)  # [num_clusters]
    cluster_idx = torch.argmin(distances).item()
    return cluster_idx


def _update_centroid(
    centroid_old: torch.Tensor,  # [head_dim]
    k_new: torch.Tensor,  # [head_dim]
    cluster_size_old: int,
) -> torch.Tensor:
    """增量更新质心.

    Formula: new_centroid = (old_centroid * size + k_new) / (size + 1)
    """
    return (centroid_old * cluster_size_old + k_new) / (cluster_size_old + 1)


def _top_p_selection(
    scores: torch.Tensor,  # [num_clusters] attention scores (softmax后)
    p: float = 0.9,
) -> torch.Tensor:
    """Top-p (nucleus) 选择，返回选中的簇索引.

    Args:
        scores: 归一化后的 attention scores
        p: 累积概率阈值

    Returns:
        selected_indices: [num_selected] 选中的簇索引
    """
    # 按分数降序排序
    sorted_scores, sorted_indices = torch.sort(scores, descending=True)

    # 计算累积概率
    cumsum_scores = torch.cumsum(sorted_scores, dim=0)

    # 找到累积概率超过 p 的位置
    mask = cumsum_scores <= p
    # 至少选择1个
    if mask.sum() == 0:
        mask[0] = True
    # 包含第一个超过 p 的点
    else:
        first_exceed = (cumsum_scores > p).nonzero(as_tuple=True)[0]
        if len(first_exceed) > 0:
            mask[first_exceed[0]] = True

    selected_indices = sorted_indices[mask]
    return selected_indices


class TorchNaivePrefillImpl(FMHAImplBase):
    """Torch Naive Prefill Attention Implementation.

    Uses PyTorch's scaled_dot_product_attention as a fallback when optimized
    kernels are not available. Processes sequences one at a time in a loop.
    """

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        """Initialize Torch Naive Prefill implementation.

        Args:
            attn_configs: Attention configuration
            attn_inputs: Attention inputs
            parallelism_config: Parallelism configuration (optional)
        """
        self.attn_configs = attn_configs
        self.attn_inputs = attn_inputs

        # Extract configuration
        self.num_heads = attn_configs.head_num
        self.num_kv_heads = attn_configs.kv_head_num
        self.head_dim = attn_configs.size_per_head
        self.scaling = 1.0 / (self.head_dim**0.5)
        self.enable_gqa = self.num_heads != self.num_kv_heads

        # Create RoPE and KV Cache write operations
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.rope_kvcache_impl = FusedRopeKVCachePrefillOpQKVOut(attn_configs)

        # Create write cache store implementation
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)

        # Create dummy fmha_params for interface compatibility (PyModelOutputs expects it)
        self.fmha_params = DummyFMHAParams()

        logging.debug(
            f"TorchNaivePrefillImpl initialized: heads={self.num_heads}, "
            f"kv_heads={self.num_kv_heads}, head_dim={self.head_dim}, gqa={self.enable_gqa}"
        )

    @classmethod
    def support(
        cls,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
    ) -> bool:
        """Check if this implementation supports the given configuration.

        Always returns True as a fallback, except for unsupported cases.

        Args:
            attn_configs: Attention configuration
            attn_inputs: Attention inputs

        Returns:
            True if supported, False otherwise
        """
        # Don't support MLA
        if attn_configs.use_mla:
            return False

        # Always support as fallback
        return True

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache],
    ) -> torch.Tensor:
        """Forward pass for prefill attention.

        Args:
            qkv: Input QKV tensor [total_tokens, (num_heads + 2*num_kv_heads) * head_dim]
            kv_cache: KV cache object (optional)

        Returns:
            Attention output [total_tokens, num_heads * head_dim]
        """
        # 1. Apply RoPE if needed
        if self.need_rope_kv_cache:
            qkv = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)

        # 2. Split QKV
        q, k, v = self._split_qkv(qkv)

        # 4. Apply write cache store (for prefill with prefix)
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        # 5. Execute attention (K, V are already complete for prefill)
        output = self._run_attention_extend(q, k, v)

        # 6. Reshape output to [total_tokens, num_heads * head_dim]
        output = output.reshape(output.shape[0], -1)

        return output

    def _perform_k_clustering_if_available(
        self,
        k: torch.Tensor,  # [total_tokens, num_kv_heads, head_dim]
        kv_cache: Optional[KVCache],
    ) -> None:
        """对 K 进行聚类（可选）用于 Decode 阶段加速.

        此方法在原始 Prefill 中调用，为 Decode 的聚类做准备。
        """
        import os

        cluster_ratio = int(os.getenv("CLUSTER_RATIO", "64"))
        kmeans_iters = int(os.getenv("KMEANS_ITERS", "20"))

        if kv_cache is None:
            return

        layer_id = kv_cache.layer_id
        batch_size = self.attn_inputs.input_lengths.size(0)
        cu_seqlens = self.attn_inputs.cu_seqlens[: batch_size + 1]

        # 按序列和 head 聚类
        for seq_idx in range(batch_size):
            start_idx = cu_seqlens[seq_idx].item()
            end_idx = cu_seqlens[seq_idx + 1].item()
            seq_len = end_idx - start_idx

            per_seq_k = k[start_idx:end_idx, :, :]  # [seq_len, num_kv_heads, head_dim]

            # 对每个 KV head 独立聚类
            for head_idx in range(per_seq_k.shape[1]):
                k_head = per_seq_k[:, head_idx, :]  # [seq_len, head_dim]

                # 计算簇数量
                num_clusters = max(1, seq_len // cluster_ratio)

                # K-Means 聚类
                centroids, labels, cluster_indices = _kmeans_clustering(
                    k_head, num_clusters, max_iters=kmeans_iters
                )

                # 计算每个簇的大小
                cluster_sizes = torch.bincount(labels, minlength=num_clusters)

                # 存储到全局缓存
                key = f"layer_{layer_id}_seq_{seq_idx}_head_{head_idx}"
                _CLUSTER_CACHE[key] = {
                    "centroids": centroids,
                    "cluster_sizes": cluster_sizes,
                    "cluster_indices": cluster_indices,
                    "seq_len": seq_len,
                }

                logging.debug(
                    f"K-Clustering (from Prefill): {key}, seq_len={seq_len}, "
                    f"num_clusters={num_clusters}, sizes={cluster_sizes.tolist()}"
                )

    def _split_qkv(
        self, qkv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split QKV tensor into query, key, value.

        Args:
            qkv: QKV tensor [total_tokens, (num_heads + 2*num_kv_heads) * head_dim]

        Returns:
            Tuple of (query, key, value) tensors
            - query: [total_tokens, num_heads, head_dim]
            - key: [total_tokens, num_kv_heads, head_dim]
            - value: [total_tokens, num_kv_heads, head_dim]
        """
        qkv = qkv.reshape(qkv.shape[0], -1)

        q, k, v = torch.split(
            qkv,
            [
                self.head_dim * self.num_heads,
                self.head_dim * self.num_kv_heads,
                self.head_dim * self.num_kv_heads,
            ],
            dim=-1,
        )

        q = q.reshape(q.shape[0], self.num_heads, self.head_dim)
        k = k.reshape(k.shape[0], self.num_kv_heads, self.head_dim)
        v = v.reshape(v.shape[0], self.num_kv_heads, self.head_dim)

        return q, k, v

    def _run_attention_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Execute prefill attention with causal masking.

        Processes each sequence separately due to varying sequence lengths.
        This is inefficient but works as a fallback.

        Args:
            q: Query tensor [total_tokens, num_heads, head_dim]
            k: Key tensor [total_tokens, num_kv_heads, head_dim]
            v: Value tensor [total_tokens, num_kv_heads, head_dim]

        Returns:
            Attention output [total_tokens, num_heads, head_dim]
        """
        # Get sequence information
        batch_size = self.attn_inputs.input_lengths.size(0)
        cu_seqlens = self.attn_inputs.cu_seqlens[: batch_size + 1]

        # Prepare output tensor
        output = torch.empty_like(q)

        # Process each sequence separately
        for seq_idx in range(batch_size):
            start_idx = cu_seqlens[seq_idx].item()
            end_idx = cu_seqlens[seq_idx + 1].item()
            seq_len = end_idx - start_idx

            # Extract per-sequence tensors
            per_seq_q = q[start_idx:end_idx, :, :]  # [seq_len, num_heads, head_dim]
            per_seq_k = k[start_idx:end_idx, :, :]  # [seq_len, num_kv_heads, head_dim]
            per_seq_v = v[start_idx:end_idx, :, :]  # [seq_len, num_kv_heads, head_dim]

            # Handle GQA: expand K, V heads to match Q heads
            if self.enable_gqa:
                # Repeat K, V heads: [seq_len, num_kv_heads, head_dim] -> [seq_len, num_heads, head_dim]
                num_groups = self.num_heads // self.num_kv_heads
                per_seq_k = per_seq_k.repeat_interleave(num_groups, dim=1)
                per_seq_v = per_seq_v.repeat_interleave(num_groups, dim=1)

            # Transpose for SDPA: [num_heads, seq_len, head_dim]
            per_seq_q = per_seq_q.movedim(0, 1)
            per_seq_k = per_seq_k.movedim(0, 1)
            per_seq_v = per_seq_v.movedim(0, 1)

            # Handle dtype mismatch (SDPA requires same dtype)
            if not (per_seq_q.dtype == per_seq_k.dtype == per_seq_v.dtype):
                per_seq_k = per_seq_k.to(per_seq_q.dtype)
                per_seq_v = per_seq_v.to(per_seq_q.dtype)

            # Execute scaled_dot_product_attention
            # Add batch dimension: [1, num_heads, seq_len, head_dim]
            per_seq_out = scaled_dot_product_attention(
                per_seq_q.unsqueeze(0),
                per_seq_k.unsqueeze(0),
                per_seq_v.unsqueeze(0),
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
                scale=self.scaling,
            ).squeeze(0)

            # Transpose back: [seq_len, num_heads, head_dim]
            per_seq_out = per_seq_out.movedim(1, 0)

            # Store result
            output[start_idx:end_idx, :, :] = per_seq_out

        return output


class TorchNaiveDecodeImpl(FMHAImplBase):
    """Torch Naive Decode Attention Implementation.

    Uses PyTorch's scaled_dot_product_attention for decode phase.
    Currently a placeholder - will be implemented in Phase 2.
    """

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        """Initialize Torch Naive Decode implementation.

        Args:
            attn_configs: Attention configuration
            attn_inputs: Attention inputs
            parallelism_config: Parallelism configuration (optional)
        """
        self.attn_configs = attn_configs
        self.attn_inputs = attn_inputs

        # Extract configuration
        self.num_heads = attn_configs.head_num
        self.num_kv_heads = attn_configs.kv_head_num
        self.head_dim = attn_configs.size_per_head
        self.scaling = 1.0 / (self.head_dim**0.5)
        self.enable_gqa = self.num_heads != self.num_kv_heads
        self.tokens_per_block = attn_configs.tokens_per_block

        # Create RoPE and KV Cache operations
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.rope_kvcache_impl = FusedRopeKVCacheDecodeOp(attn_configs)

        # Create write cache store implementation
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)

        # Create dummy fmha_params for interface compatibility (PyModelOutputs expects it)
        self.fmha_params = DummyFMHAParams()

        logging.debug(
            f"TorchNaiveDecodeImpl initialized: heads={self.num_heads}, "
            f"kv_heads={self.num_kv_heads}, head_dim={self.head_dim}, gqa={self.enable_gqa}"
        )

    @classmethod
    def support(
        cls,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
    ) -> bool:
        """Check if this implementation supports the given configuration.

        Always returns True as a fallback, except for unsupported cases.

        Args:
            attn_configs: Attention configuration
            attn_inputs: Attention inputs

        Returns:
            True if supported, False otherwise
        """
        # Don't support MLA
        if attn_configs.use_mla:
            return False

        # Always support as fallback
        return True

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache],
    ) -> torch.Tensor:
        """Forward pass for decode attention.

        Args:
            qkv: Input QKV tensor [batch_size, (num_heads + 2*num_kv_heads) * head_dim]
            kv_cache: KV cache object (required for decode)

        Returns:
            Attention output [batch_size, num_heads * head_dim]
        """
        # 1. Apply RoPE if needed
        # NOTE: Decode RoPE writes K,V to cache directly and only returns Q
        if self.need_rope_kv_cache:
            q = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)

            # RoPE may return Q in different shapes, normalize to [batch, num_heads, head_dim]
            if q.ndim == 2:
                # 2D: [batch, num_heads * head_dim] -> reshape to 3D
                q = q.reshape(q.shape[0], self.num_heads, self.head_dim)
            elif q.ndim == 3:
                # Already 3D [batch, num_heads, head_dim] - no change needed
                pass
            else:
                raise ValueError(f"Unexpected Q shape from RoPE: {q.shape}")
        else:
            # No RoPE: split QKV manually (though this path is unlikely for decode)
            q, k, v = self._split_qkv(qkv)

        # 4. Apply write cache store (for decode with new tokens)
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        # 5. Read complete K, V from cache (including history)
        k_full, v_full = self._read_kv_from_cache(kv_cache)
        logging.info(
            f"[Decode] q shape: {q.shape}, k_full: {k_full.shape}, v_full: {v_full.shape}"
        )
        logging.info(
            f"[Decode] q dtype: {q.dtype}, k dtype: {k_full.dtype}, v dtype: {v_full.dtype}"
        )

        # 6. Execute decode attention (no causal mask needed - single query token)
        output = self._run_attention_decode(q, k_full, v_full)
        logging.info(f"[Decode] output shape: {output.shape}")

        # 7. Reshape output to [batch_size, num_heads * head_dim]
        output = output.reshape(output.shape[0], -1)

        return output

    def _split_qkv(
        self, qkv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split QKV tensor into query, key, value.

        Args:
            qkv: QKV tensor [batch_size, (num_heads + 2*num_kv_heads) * head_dim]

        Returns:
            Tuple of (query, key, value) tensors
            - query: [batch_size, num_heads, head_dim]
            - key: [batch_size, num_kv_heads, head_dim]
            - value: [batch_size, num_kv_heads, head_dim]
        """
        qkv = qkv.reshape(qkv.shape[0], -1)

        # Debug: check dimensions
        expected_size = self.head_dim * (self.num_heads + 2 * self.num_kv_heads)
        actual_size = qkv.shape[-1]
        if expected_size != actual_size:
            logging.error(
                f"QKV size mismatch in {self.__class__.__name__}: "
                f"expected {expected_size} (heads={self.num_heads}, kv_heads={self.num_kv_heads}, "
                f"head_dim={self.head_dim}), got {actual_size}"
            )
            # Adjust num_heads based on actual size
            # This might happen if RoPE changed the format
            actual_qkv_heads = actual_size // self.head_dim
            if actual_qkv_heads < 2 * self.num_kv_heads:
                logging.error(f"Cannot split: not enough space for K and V")
                raise ValueError(
                    f"QKV size {actual_size} too small for kv_heads={self.num_kv_heads}"
                )

        q, k, v = torch.split(
            qkv,
            [
                self.head_dim * self.num_heads,
                self.head_dim * self.num_kv_heads,
                self.head_dim * self.num_kv_heads,
            ],
            dim=-1,
        )

        q = q.reshape(q.shape[0], self.num_heads, self.head_dim)
        k = k.reshape(k.shape[0], self.num_kv_heads, self.head_dim)
        v = v.reshape(v.shape[0], self.num_kv_heads, self.head_dim)

        return q, k, v

    def _read_kv_from_cache(
        self, kv_cache: KVCache
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read complete K, V from paged KV cache (including history).

        Args:
            kv_cache: KV cache object containing paged cache data

        Returns:
            Tuple of (k_full, v_full) tensors
            - k_full: [batch_size, total_seq_len, num_kv_heads, head_dim]
            - v_full: [batch_size, total_seq_len, num_kv_heads, head_dim]
        """
        # Get batch size
        batch_size = self.attn_inputs.input_lengths.size(0)

        # Get real sequence lengths using FlashInferMlaAttnParams
        # This correctly computes kvlen from sequence_lengths + 1 in decode mode
        from rtp_llm.ops.compute_ops import fill_mla_params

        params = fill_mla_params(
            (
                self.attn_inputs.prefix_lengths
                if hasattr(self.attn_inputs, "prefix_lengths")
                else torch.tensor([], dtype=torch.int32)
            ),
            self.attn_inputs.sequence_lengths,
            self.attn_inputs.input_lengths,
            self.attn_inputs.kv_cache_block_id_host,
            self.tokens_per_block,
        )

        # kvlen contains the REAL sequence lengths (including current token in decode mode)
        sequence_lengths = params.kvlen_h[:batch_size]
        max_seq_len = sequence_lengths.max().item()

        logging.info(
            f"[_read_kv_from_cache] batch_size={batch_size}, real_seq_lengths={sequence_lengths.tolist()}, max_seq_len={max_seq_len}"
        )

        # Get KV cache tensor and reshape if needed
        # kv_cache_base may be 2D [num_blocks, kv_block_stride_elems] and needs reshaping to 5D
        kv_cache_base = kv_cache.kv_cache_base
        layer_id = kv_cache.layer_id

        # Reshape to 5D: [num_blocks, 2, num_kv_heads, tokens_per_block, head_dim]
        if kv_cache_base.ndim == 2:
            block_num = kv_cache_base.shape[0]
            expected_elems = (
                2 * self.num_kv_heads * self.tokens_per_block * self.head_dim
            )
            kv_cache_tensor = kv_cache_base[:, :expected_elems].reshape(
                block_num,
                2,
                self.num_kv_heads,
                self.tokens_per_block,
                self.head_dim,
            )
        else:
            kv_cache_tensor = kv_cache_base

        # Get block indices for each sequence
        # Shape: [batch_size, max_blocks_per_seq]
        block_indices = self.attn_inputs.kv_cache_block_id_host[:batch_size, :]

        # Prepare output tensors
        k_full = torch.zeros(
            batch_size,
            max_seq_len,
            self.num_kv_heads,
            self.head_dim,
            dtype=kv_cache_tensor.dtype,
            device=kv_cache_tensor.device,
        )
        v_full = torch.zeros(
            batch_size,
            max_seq_len,
            self.num_kv_heads,
            self.head_dim,
            dtype=kv_cache_tensor.dtype,
            device=kv_cache_tensor.device,
        )

        # Read K, V for each sequence
        for batch_idx in range(batch_size):
            seq_len = sequence_lengths[batch_idx].item()
            num_blocks = (seq_len + self.tokens_per_block - 1) // self.tokens_per_block

            # Collect K, V from blocks
            for block_idx in range(num_blocks):
                block_id = block_indices[batch_idx, block_idx].item()

                # Calculate token range for this block
                start_token = block_idx * self.tokens_per_block
                end_token = min(start_token + self.tokens_per_block, seq_len)
                block_token_count = end_token - start_token

                # Read K, V from cache
                # kv_cache_tensor shape: [num_blocks, 2, num_kv_heads, tokens_per_block, head_dim]
                k_block = kv_cache_tensor[
                    block_id, 0, :, :block_token_count, :
                ]  # [kv_heads, block_tokens, head_dim]
                v_block = kv_cache_tensor[
                    block_id, 1, :, :block_token_count, :
                ]  # [kv_heads, block_tokens, head_dim]

                # Store in output tensors
                # Transpose to [block_tokens, kv_heads, head_dim]
                k_full[batch_idx, start_token:end_token, :, :] = k_block.transpose(0, 1)
                v_full[batch_idx, start_token:end_token, :, :] = v_block.transpose(0, 1)

        return k_full, v_full

    def _run_attention_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Execute decode attention.

        In decode phase, each query is a single token attending to all previous tokens.

        Args:
            q: Query tensor [batch_size, num_heads, head_dim]
            k: Key tensor [batch_size, total_seq_len, num_kv_heads, head_dim]
            v: Value tensor [batch_size, total_seq_len, num_kv_heads, head_dim]

        Returns:
            Attention output [batch_size, num_heads, head_dim]
        """
        batch_size = q.shape[0]

        # Handle GQA: expand K, V heads to match Q heads
        if self.enable_gqa:
            # k, v: [batch_size, seq_len, num_kv_heads, head_dim]
            # Need to expand to [batch_size, seq_len, num_heads, head_dim]
            num_groups = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(num_groups, dim=2)
            v = v.repeat_interleave(num_groups, dim=2)

        # Reshape for SDPA
        # q: [batch_size, num_heads, head_dim] -> [batch_size, num_heads, 1, head_dim]
        # k, v: [batch_size, seq_len, num_heads, head_dim] -> [batch_size, num_heads, seq_len, head_dim]
        q = q.unsqueeze(2)  # Add seq_len dimension
        k = k.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        v = v.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]

        # Handle dtype mismatch (SDPA requires same dtype)
        if not (q.dtype == k.dtype == v.dtype):
            k = k.to(q.dtype)
            v = v.to(q.dtype)

        # Execute scaled_dot_product_attention
        # No causal mask needed - single query attends to all past tokens
        output = scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,  # No causal mask for decode
            scale=self.scaling,
        )

        # output shape: [batch_size, num_heads, 1, head_dim]
        # Squeeze to remove seq_len dimension: [batch_size, num_heads, head_dim]
        output = output.squeeze(2)

        return output


# ============================================================================
# K-Clustering Enhanced Implementations
# ============================================================================


class TorchNaiveClusteredPrefillImpl(TorchNaivePrefillImpl):
    """带 K 聚类的 Prefill Attention 实现.

    继承自 TorchNaivePrefillImpl，在 Prefill 阶段对 K 进行聚类，
    存储质心、簇大小和 token 索引，供 Decode 阶段使用。
    """

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        """初始化带聚类的 Prefill 实现."""
        # 调用父类初始化
        super().__init__(attn_configs, attn_inputs, parallelism_config)

        # 聚类配置（从环境变量读取）
        import os

        self.cluster_ratio = int(os.getenv("CLUSTER_RATIO", "64"))
        self.kmeans_iters = int(os.getenv("KMEANS_ITERS", "20"))

        # NEW: 存储每个 (layer, seq, head) 的 IncrementalKMeans 实例
        self.incremental_models = {}  # key -> IncrementalKMeans

        logging.debug(
            f"TorchNaiveClusteredPrefillImpl initialized: "
            f"ratio={self.cluster_ratio}, kmeans_iters={self.kmeans_iters}"
        )

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        """支持检查：同父类."""
        return super(TorchNaiveClusteredPrefillImpl, cls).support(
            attn_configs, attn_inputs
        )

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache],
    ) -> torch.Tensor:
        """Forward pass: 先做聚类，再执行 attention."""

        # 1. Apply RoPE if needed
        if self.need_rope_kv_cache:
            qkv = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)

        # 2. Split QKV
        q, k, v = self._split_qkv(qkv)

        # 3. K Clustering (NEW) - 在写入 cache 之前做聚类
        self._perform_k_clustering(k, kv_cache)

        # 5. Apply write cache store
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        # 6. Execute attention (同父类，使用完整的 K)
        output = self._run_attention_extend(q, k, v)

        # 7. Reshape output
        output = output.reshape(output.shape[0], -1)

        return output

    def _perform_k_clustering(
        self,
        k: torch.Tensor,  # [total_tokens, num_kv_heads, head_dim]
        kv_cache: Optional[KVCache],
    ) -> None:
        """对 K 进行聚类并存储结果（使用 IncrementalKMeans）.

        Args:
            k: Key tensor
            kv_cache: KV cache object (用于获取 layer_id)
        """
        layer_id = kv_cache.layer_id if kv_cache is not None else 0
        batch_size = self.attn_inputs.input_lengths.size(0)
        cu_seqlens = self.attn_inputs.cu_seqlens[: batch_size + 1]

        # 按序列和 head 聚类
        for seq_idx in range(batch_size):
            start_idx = cu_seqlens[seq_idx].item()
            end_idx = cu_seqlens[seq_idx + 1].item()
            seq_len = end_idx - start_idx

            per_seq_k = k[start_idx:end_idx, :, :]  # [seq_len, num_kv_heads, head_dim]

            # 对每个 KV head 独立聚类
            for head_idx in range(per_seq_k.shape[1]):
                k_head = per_seq_k[:, head_idx, :]  # [seq_len, head_dim]

                # 计算簇数量
                num_clusters = max(1, seq_len // self.cluster_ratio)

                # 使用原有的批量聚类初始化质心
                centroids, labels, cluster_indices = _kmeans_clustering(
                    k_head, num_clusters, max_iters=self.kmeans_iters
                )
                cluster_sizes = torch.bincount(labels, minlength=num_clusters)

                # NEW: 创建 IncrementalKMeans 实例
                model = IncrementalKMeans(
                    n_clusters=num_clusters,
                    dim=self.head_dim,
                    device=k_head.device,
                    dtype=k_head.dtype,
                )
                model.init_centroids(centroids)

                # 将初始数据加入统计（重要！）
                model.add_points(k_head, update_centroids=False)

                # 存储模型和辅助信息
                key = f"layer_{layer_id}_seq_{seq_idx}_head_{head_idx}"
                self.incremental_models[key] = model

                # 仍然保存到 _CLUSTER_CACHE 供其他组件使用
                _CLUSTER_CACHE[key] = {
                    "centroids": centroids,
                    "cluster_sizes": cluster_sizes,
                    "cluster_indices": cluster_indices,
                    "seq_len": seq_len,
                    "model": model,  # NEW: 保存模型引用
                }

                logging.debug(
                    f"K-Clustering (IncrementalKMeans): {key}, "
                    f"seq_len={seq_len}, num_clusters={num_clusters}"
                )


class TorchNaiveClusteredDecodeImpl(TorchNaiveDecodeImpl):
    """带 K 聚类加速的 Decode Attention 实现.

    继承自 TorchNaiveDecodeImpl，使用 Prefill 阶段的聚类信息，
    通过 Q @ centroids + top_p 选择 + Full Attention 来加速计算。
    """

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        """初始化带聚类的 Decode 实现."""
        super().__init__(attn_configs, attn_inputs, parallelism_config)

        # 聚类配置
        import os

        self.top_p = float(os.getenv("CLUSTER_TOP_P", "0.9"))

        logging.debug(f"TorchNaiveClusteredDecodeImpl initialized: top_p={self.top_p}")

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        """支持检查：同父类."""
        return super(TorchNaiveClusteredDecodeImpl, cls).support(
            attn_configs, attn_inputs
        )

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache],
    ) -> torch.Tensor:
        """Forward pass: 使用聚类加速 attention."""
        logging.info(
            f"[ClusteredDecode] forward: input qkv shape={qkv.shape}, need_rope={self.need_rope_kv_cache}"
        )

        # 1. Apply RoPE if needed
        # NOTE: Decode RoPE may write K,V to cache directly and only return Q
        if self.need_rope_kv_cache:
            q = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)

        # 4. Apply write cache store
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        # 5. Update clustering: 将新 K 分配到簇并更新质心
        k_full_temp, _ = self._read_kv_from_cache(kv_cache)
        k = k_full_temp[:, -1:, :, :]  # Get only the last token (new K)
        k = k.squeeze(1)  # Remove seq dimension: [batch, kv_heads, head_dim]

        self._update_clustering(k, kv_cache)

        # 6. Read complete K, V from cache (including history)
        k_full, v_full = self._read_kv_from_cache(kv_cache)

        # 7. Execute clustered decode attention (NEW)
        output = self._run_clustered_attention_decode(q, k_full, v_full, kv_cache)

        # 8. Reshape output
        output = output.reshape(output.shape[0], -1)

        return output

    def _update_clustering(
        self,
        k_new: torch.Tensor,  # [batch_size, num_kv_heads, head_dim]
        kv_cache: Optional[KVCache],
    ) -> None:
        """使用 IncrementalKMeans 将新 K 增量更新到聚类（NEW 实现）.

        Args:
            k_new: 新生成的 Key tensor
            kv_cache: KV cache object
        """
        if kv_cache is None:
            return

        layer_id = kv_cache.layer_id
        batch_size = k_new.shape[0]

        for batch_idx in range(batch_size):
            for head_idx in range(k_new.shape[1]):
                key = f"layer_{layer_id}_seq_{batch_idx}_head_{head_idx}"

                if key not in _CLUSTER_CACHE:
                    # 没有聚类信息，跳过（可能是新序列）
                    exit(0)
                    # continue

                cluster_info = _CLUSTER_CACHE[key]

                # NEW: 使用 IncrementalKMeans
                if "model" in cluster_info:
                    model = cluster_info["model"]
                    k_single = k_new[batch_idx, head_idx, :].unsqueeze(
                        0
                    )  # [1, head_dim]

                    # 增量添加新点并更新质心（一行代码完成！）
                    label = model.add_points(k_single, update_centroids=True)

                    # 更新 cache 中的质心和统计信息
                    cluster_info["centroids"] = model.get_centroids()
                    _, counts = model.get_statistics()
                    cluster_info["cluster_sizes"] = counts

                    # 更新簇的 token 索引
                    cluster_idx = label.item()
                    new_token_idx = cluster_info["seq_len"]
                    cluster_info["cluster_indices"][cluster_idx].append(new_token_idx)
                    cluster_info["seq_len"] += 1

                    logging.debug(
                        f"Update clustering (IncrementalKMeans): {key}, "
                        f"assigned to cluster {cluster_idx}, "
                        f"new_size={counts[cluster_idx].item()}"
                    )
                else:
                    # FALLBACK: 使用旧的手动实现（兼容性）
                    centroids = cluster_info["centroids"]
                    sizes = cluster_info["cluster_sizes"]
                    indices = cluster_info["cluster_indices"]

                    k_single = k_new[batch_idx, head_idx, :]
                    cluster_idx = _assign_to_cluster(k_single, centroids)
                    centroids[cluster_idx] = _update_centroid(
                        centroids[cluster_idx], k_single, sizes[cluster_idx].item()
                    )
                    sizes[cluster_idx] += 1

                    new_token_idx = cluster_info["seq_len"]
                    indices[cluster_idx].append(new_token_idx)
                    cluster_info["seq_len"] += 1

                    logging.warning(
                        f"Fallback to manual update for {key} "
                        f"(IncrementalKMeans model not found)"
                    )

    def _run_clustered_attention_decode(
        self,
        q: torch.Tensor,  # [batch_size, num_heads, head_dim]
        k_full: torch.Tensor,  # [batch_size, total_seq_len, num_kv_heads, head_dim]
        v_full: torch.Tensor,  # [batch_size, total_seq_len, num_kv_heads, head_dim]
        kv_cache: Optional[KVCache],
    ) -> torch.Tensor:
        """使用聚类加速的 Decode Attention.

        流程:
        1. Q @ centroids 计算 attention score
        2. Top-p 选择重要的簇
        3. 对选中簇的 tokens 做 Full Attention

        Returns:
            output: [batch_size, num_heads, head_dim]
        """
        batch_size = q.shape[0]
        layer_id = kv_cache.layer_id if kv_cache is not None else 0

        # GQA handling
        if self.enable_gqa:
            num_groups = self.num_heads // self.num_kv_heads
            k_full = k_full.repeat_interleave(num_groups, dim=2)
            v_full = v_full.repeat_interleave(num_groups, dim=2)

        output = torch.empty_like(q)

        # 按 batch 和 head 处理
        for batch_idx in range(batch_size):
            for head_idx in range(q.shape[1]):
                # 将 Q head index 映射到 KV head index（GQA 场景）
                if self.enable_gqa:
                    num_groups = self.num_heads // self.num_kv_heads
                    kv_head_idx = head_idx // num_groups
                else:
                    kv_head_idx = head_idx

                key = f"layer_{layer_id}_seq_{batch_idx}_head_{kv_head_idx}"

                # 如果没有聚类信息，fallback 到 Full Attention
                if key not in _CLUSTER_CACHE:
                    # output[batch_idx, head_idx, :] = self._full_attention_single(
                    #     q[batch_idx, head_idx, :],
                    #     k_full[batch_idx, :, head_idx, :],
                    #     v_full[batch_idx, :, head_idx, :]
                    # )
                    # continue
                    exit(0)

                cluster_info = _CLUSTER_CACHE[key]

                # NEW: 优先从 IncrementalKMeans 模型获取质心
                if "model" in cluster_info:
                    centroids = cluster_info["model"].get_centroids()
                else:
                    centroids = cluster_info["centroids"]

                cluster_indices = cluster_info["cluster_indices"]

                q_single = q[batch_idx, head_idx, :]  # [head_dim]

                # Step 1: Q @ centroids 计算 attention score
                scores = (
                    torch.matmul(q_single, centroids.T) * self.scaling
                )  # [num_clusters]

                # 加上簇大小的对数，让大簇有更高的权重
                cluster_sizes = cluster_info["cluster_sizes"]
                scores = scores + torch.log(cluster_sizes.float() + 1e-8)  # 避免 log(0)

                scores = torch.softmax(scores, dim=0)

                # Step 2: Top-p 选择簇
                selected_cluster_ids = _top_p_selection(scores, self.top_p)

                # Step 3: 收集选中簇的所有 token 索引
                selected_token_ids = []
                for cluster_id in selected_cluster_ids:
                    selected_token_ids.extend(cluster_indices[cluster_id.item()])

                if len(selected_token_ids) == 0:
                    # 没有选中任何 token，使用零向量或 fallback
                    output[batch_idx, head_idx, :] = 0.0
                    logging.warning(f"No tokens selected for {key}, using zero output")
                    continue

                # 去重并排序
                selected_token_ids = sorted(set(selected_token_ids))

                # Step 4: 对选中的 tokens 做 Full Attention
                selected_k = k_full[
                    batch_idx, selected_token_ids, head_idx, :
                ]  # [num_selected, head_dim]
                selected_v = v_full[batch_idx, selected_token_ids, head_idx, :]

                # 打印聚类统计信息
                total_tokens = k_full.shape[1]
                total_clusters = centroids.shape[0]
                selected_tokens_count = len(selected_token_ids)
                selected_clusters_count = len(selected_cluster_ids)

                logging.info(
                    f"[Clustering Stats] {key}: "
                    f"Selected {selected_clusters_count}/{total_clusters} clusters ({selected_clusters_count/total_clusters*100:.1f}%), "
                    f"Selected {selected_tokens_count}/{total_tokens} tokens ({selected_tokens_count/total_tokens*100:.1f}%)"
                )

                output[batch_idx, head_idx, :] = self._full_attention_single(
                    q_single, selected_k, selected_v
                )

                logging.debug(
                    f"Clustered attention: {key}, selected {len(selected_token_ids)} / "
                    f"{k_full.shape[1]} tokens ({len(selected_token_ids)/k_full.shape[1]*100:.1f}%)"
                )

        return output

    def _full_attention_single(
        self,
        q: torch.Tensor,  # [head_dim]
        k: torch.Tensor,  # [seq_len, head_dim]
        v: torch.Tensor,  # [seq_len, head_dim]
    ) -> torch.Tensor:
        """单个 query 的 Full Attention.

        Returns:
            output: [head_dim]
        """
        # Reshape for SDPA
        q = q.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, head_dim]
        k = k.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
        v = v.unsqueeze(0).unsqueeze(0)

        # Handle dtype mismatch
        if not (q.dtype == k.dtype == v.dtype):
            k = k.to(q.dtype)
            v = v.to(q.dtype)

        output = scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=self.scaling,
        ).squeeze()

        return output
