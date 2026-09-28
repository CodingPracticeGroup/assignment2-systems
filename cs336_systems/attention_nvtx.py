"""给 self-attention 内部插 NVTX 区间。

用途：讲义 **Problem (nsys_profile) 的 (e)** ——
    比较 self-attention 里 **softmax** 与 **矩阵乘** 的 runtime，
    再看这个 runtime 差距和它们的 **FLOPs 差距** 是否成比例。

讲义的做法（§2.1.4 末）是猴子补丁：

    cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

`baseline.py` 的 `--annotate-attention` 就是干这个的。

注意：基础实现抄自 `cs336_basics.model.scaled_dot_product_attention`，只加了 `nvtx.range`，
**计算逻辑一字未改**——否则测出来的就不是原实现的性能了。
"""

from __future__ import annotations

import math

import torch
import torch.cuda.nvtx as nvtx
from einops import einsum

from cs336_basics.nn_utils import softmax

# 三个区间的名字（报表里按这个过滤）
RANGE_QK = "attn: QK^T matmul"
RANGE_SOFTMAX = "attn: softmax"
RANGE_PV = "attn: PV matmul"
ALL_RANGES = (RANGE_QK, RANGE_SOFTMAX, RANGE_PV)


def annotated_scaled_dot_product_attention(K, Q, V, mask=None):
    """与 cs336_basics.model.scaled_dot_product_attention 等价，只是分了三段 NVTX。"""
    d_k = K.shape[-1]

    with nvtx.range(RANGE_QK):
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

    if mask is not None:
        attention_scores = torch.where(mask, attention_scores, float("-inf"))

    with nvtx.range(RANGE_SOFTMAX):
        attention_weights = softmax(attention_scores, dim=-1)

    with nvtx.range(RANGE_PV):
        out = einsum(attention_weights, V, "... query key, ... key d_v -> ... query d_v")

    return out


def attention_flops(batch: int, seq_len: int, d_model: int, num_heads: int) -> dict[str, float]:
    """一次 forward 里、**全部层**的 self-attention 各部件 FLOPs（乘加算 2 次）。

    记 d_head = d_model / num_heads：
      QK^T : 每个 head 2*b*s*s*d_head，共 a 个 head → 2*b*s^2*d_model
      PV   : 同上                              → 2*b*s^2*d_model
      softmax: 每元素约 5 次操作（max / sub / exp / sum / div），作用在 a*b*s*s 个元素上
               → 5*a*b*s^2
    """
    a, b, s, d = num_heads, batch, seq_len, d_model
    return {
        RANGE_QK: 2 * b * s * s * d,
        RANGE_PV: 2 * b * s * s * d,
        RANGE_SOFTMAX: 5 * a * b * s * s,
    }
