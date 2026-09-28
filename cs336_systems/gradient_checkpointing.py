"""梯度检查点（activation checkpointing）策略对比 —— assignment 2 §3.2 / (f) 的 (a)(b)。

讲义 (a)：N 个相同 block 顺序堆叠，checkpoint 可任意嵌套，忽略计算开销时
什么策略最小化峰值激活显存？
讲义 (b)：只允许一层重算（不能嵌套）时，xl @ ctx2048 的最佳 block size 是多少？

本模块把三种策略放进同一个 `forward` 里，方便直接对比峰值显存与单步耗时：

    none       : 不 checkpoint，N 层的残差全部留存   → O(N) 峰值
    block      : 每 k 层包一个 checkpoint            → N/k 个 checkpoint + k 层残差
    recursive  : 二分递归嵌套 checkpoint             → O(log N) 个 checkpoint + 1 层残差

**关键机制**（决定了 recursive 为什么能到 O(log N)）：外层 checkpoint 在 forward 时
把内层的保存行为一并抑制掉，于是内层 checkpoint 的输入在 forward 阶段**不会留存**，
只在外层反向重算时才被重新创建、用完即释放。所以任一时刻存活的 checkpoint 数
= 递归深度 = O(log N)，而不是 N-1。`--verify-nesting` 会实测验证这条机制。

用法：

    # 1) 复现讲义 §3.2 的"单个 block 存了多少残差"（xl @ ctx2048 → 讲义报 3651.31 MiB）
    uv run python -m cs336_systems.gradient_checkpointing residual \
        --model-size xl --context-length 2048 --batch-size 4 --dtype fp32

    # 2) 扫描策略 / block size，打印峰值显存与单步耗时
    uv run python -m cs336_systems.gradient_checkpointing scan \
        --model-size xl --context-length 2048 --batch-size 4 --dtype bf16 \
        --block-sizes 1 2 4 8 16 32 --with-recursive
"""

from __future__ import annotations

import argparse
import gc
import os
import sys

# 必须在任何 CUDA 调用之前设置：xl 这种"参数+梯度就占 12.7/15.3 GiB"的边缘配置，
# 默认 allocator 的碎片会直接导致 OOM（实测 reserved 15.29 GiB 仍失败）。
# expandable_segments 让段可以增长，实测把 full step 从 OOM 变成能跑（峰值 14.85 GiB）。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import timeit
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

_BASICS = Path(__file__).resolve().parent.parent / "cs336-basics"
if str(_BASICS) not in sys.path:
    sys.path.insert(0, str(_BASICS))

from cs336_basics.model import BasicsTransformerLM  # noqa: E402

MiB = 1024 * 1024
GiB = 1024**3
VOCAB_SIZE = 10000
MODEL_SIZES = {
    "small": (768, 3072, 12, 12),
    "medium": (1024, 4096, 24, 16),
    "large": (1280, 5120, 36, 20),
    "xl": (2560, 10240, 32, 32),
}


# --------------------------------------------------------------------------------------
# 策略
# --------------------------------------------------------------------------------------
def _run_segment(seg: list, h: torch.Tensor) -> torch.Tensor:
    for blk in seg:
        h = blk(h)
    return h


def _recursive(blocks: list, h: torch.Tensor) -> torch.Tensor:
    """二分递归嵌套 checkpoint：内部节点各包一层，叶子（单个 block）直接跑。

    叶子不包 checkpoint 是有意的：叶子外面已经有一层 checkpoint 抑制保存了，
    再包一层只会多存一个输入张量、不减少重算量。
    """
    if len(blocks) == 1:
        return blocks[0](h)
    mid = len(blocks) // 2
    left, right = blocks[:mid], blocks[mid:]
    h = checkpoint(lambda inp, bl=left: _recursive(bl, inp), h, use_reentrant=False)
    h = checkpoint(lambda inp, bl=right: _recursive(bl, inp), h, use_reentrant=False)
    return h


def forward_hidden(model, x: torch.Tensor, strategy: str, block_size: int) -> torch.Tensor:
    """跑完 embedding + 所有 block，返回进入 ln_final 之前的 hidden state。"""
    h = model.token_embeddings(x)
    blocks = list(model.layers)
    n = len(blocks)

    if strategy == "none":
        h = _run_segment(blocks, h)
    elif strategy == "block":
        for i in range(0, n, block_size):
            seg = blocks[i : i + block_size]
            h = checkpoint(lambda inp, s=seg: _run_segment(s, inp), h, use_reentrant=False)
    elif strategy == "recursive":
        h = _recursive(blocks, h)
    else:  # pragma: no cover
        raise ValueError(f"未知策略 {strategy}")
    return h


def model_dims(args) -> tuple[int, int, int, int]:
    """(d_model, d_ff, n_layers, n_heads)，--num-heads 可覆盖 Table 1 的值。

    为什么要能覆盖：讲义 §3.2 的实验片段用的是 num_heads=16，而 Table 1 的 xl 是 32；
    (b) 说的 "the xl model config ... as above" 指的是那个片段，所以主口径取 16。
    """
    d_model, d_ff, n_layers, n_heads = MODEL_SIZES[args.model_size]
    if getattr(args, "num_heads", None):
        n_heads = args.num_heads
    return d_model, d_ff, n_layers, n_heads


def build(args, *, train: bool = True) -> tuple:
    d_model, d_ff, n_layers, n_heads = model_dims(args)
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    # 直接用目标 dtype 构造参数（set_default_dtype），**不要**先建 fp32 再 .to(bf16)：
    # 后者会瞬时刻出一份 fp32 全量参数，实测把建模型的峰值顶到 12.87 GiB，
    # 之后 allocator 的 reserved 池就一直虚高，xl 这种边缘配置会因此 OOM。
    prev = torch.get_default_dtype()
    if dtype != torch.float32:
        torch.set_default_dtype(dtype)
    try:
        model = BasicsTransformerLM(
            vocab_size=VOCAB_SIZE,
            context_length=args.context_length,
            d_model=d_model,
            num_layers=n_layers,
            num_heads=n_heads,
            d_ff=d_ff,
        ).to(args.device)
    finally:
        torch.set_default_dtype(prev)
    if not train:
        model.eval()
    if getattr(args, "compile_blocks", False):
        # 逐 block 融合（讲义 §3.2 的做法：torch.compile(block, fullgraph=True)）。
        # 融合把一批细粒度中间量合成 unitary 算子，残差从 3543.88 → 1826.19 MiB（bf16 ctx2048），
        # 是本机唯一能让 xl full step 装进 16 GB 的关键。
        model.layers = torch.nn.ModuleList(
            [torch.compile(b, fullgraph=True) for b in model.layers]
        )
    x = torch.randint(0, VOCAB_SIZE, (args.batch_size, args.context_length), device=args.device)
    y = torch.randint(0, VOCAB_SIZE, (args.batch_size, args.context_length), device=args.device)
    return model, x, y


def one_step(model, x, y, args) -> torch.Tensor:
    # 先释放上一步的梯度。真实训练循环在末尾会 optimizer.zero_grad(set_to_none=True)，
    # 所以下一步的 forward 开始时 .grad 是 None。这里不做的话，xl 的 6.35 GiB 梯度
    # 会在下一步 forward 期间仍然常驻，直接把一个"本来刚好装下"的配置顶成 OOM。
    model.zero_grad(set_to_none=True)
    hidden = forward_hidden(model, x, args.strategy, args.block_size)
    logits = model.lm_head(model.ln_final(hidden))
    loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
    if args.mode != "forward":
        loss.backward()
    return loss


# --------------------------------------------------------------------------------------
# 子命令：residual —— 单个 block 存了多少残差（复现讲义 §3.2 的 hook 实验）
# --------------------------------------------------------------------------------------
def cmd_residual(args) -> int:
    """只建**一个** block（讲义 §3.2 就是这么做的）。

    特意不建整个模型：xl 的参数就有 12.69 GiB，再加一个 block 的 5.9 GiB 残差
    会直接 OOM——而这里要测的只是单层残差，与其它 31 层无关。
    """
    from cs336_basics.model import RotaryEmbedding, TransformerBlock

    d_model, d_ff, n_layers, n_heads = model_dims(args)
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    block = TransformerBlock(
        d_model=d_model,
        d_ff=d_ff,
        num_heads=n_heads,
        positional_encoder=RotaryEmbedding(dim=d_model // n_heads, context_length=args.context_length),
    ).to(args.device, dtype=dtype)

    saved = 0
    counts: dict[tuple, int] = {}

    def pack(t):
        nonlocal saved
        if isinstance(t, torch.nn.Parameter):  # 参数不算（讲义同样跳过，避免重复计数）
            return t
        saved += t.numel() * t.element_size()
        key = (tuple(t.shape), str(t.dtype).replace("torch.", ""))
        counts[key] = counts.get(key, 0) + 1
        return t

    def unpack(t):
        return t

    if args.compile:
        # 讲义 §3.2 的片段是 torch.compile(block, fullgraph=True)：算子融合会把
        # 一堆"过度细粒度"的中间量合成一个 unitary 算子，残差数因此大幅下降。
        block = torch.compile(block, fullgraph=True)

    h = torch.randn(
        (args.batch_size, args.context_length, d_model), device=args.device, dtype=dtype, requires_grad=True
    )
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        y = block(h)
    print(
        f"单个 TransformerBlock 为反向保存的残差: {saved / MiB:.2f} MiB "
        f"（{args.model_size} ctx={args.context_length} batch={args.batch_size} {args.dtype}）"
    )
    print(f"  即 {saved / GiB:.3f} GiB/层；{n_layers} 层全留存 = {saved * n_layers / GiB:.1f} GiB")
    print("  按形状归因（前 8 大）:")
    for (shape, dt), c in sorted(counts.items(), key=lambda kv: -kv[1] * torch.tensor(kv[0][0]).prod())[:8]:
        n = 1
        for s in shape:
            n *= s
        esz = 4 if dt == "float32" else 2
        print(f"    {str(shape):>24} {dt:>8} ×{c}  = {c * n * esz / MiB:8.2f} MiB")
    del block, h, y
    gc.collect()
    torch.cuda.empty_cache()
    return 0


# --------------------------------------------------------------------------------------
# 子命令：scan —— 扫描策略 / block size，测峰值显存与单步耗时
# --------------------------------------------------------------------------------------
def measure(model, x, y, args, *, warmup: int, steps: int) -> tuple[float, float, float]:
    for _ in range(warmup):
        one_step(model, x, y, args)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(steps):
        torch.cuda.synchronize()
        t0 = timeit.default_timer()
        one_step(model, x, y, args)
        torch.cuda.synchronize()
        times.append(timeit.default_timer() - t0)

    peak = torch.cuda.max_memory_allocated() / GiB
    t = torch.tensor(times, dtype=torch.float64)
    mean_ms = t.mean().item() * 1e3
    std_ms = t.std(unbiased=True).item() * 1e3 if len(times) > 1 else float("nan")
    return peak, mean_ms, std_ms


def cmd_scan(args) -> int:
    model, x, y = build(args)
    d_model, d_ff, n_layers, n_heads = model_dims(args)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"{args.model_size}: d_model={d_model} d_ff={d_ff} layers={n_layers} heads={n_heads} "
        f"params={n_params / 1e9:.3f}B ({n_params * model.layers[0].attn.q_proj.weight.element_size() / GiB:.2f} GiB)"
    )
    print(f"ctx={args.context_length} batch={args.batch_size} dtype={args.dtype} mode={args.mode}")
    print()

    rows: list[tuple[str, float, float, float]] = []
    plans: list[tuple[str, int]] = []
    # recursive 放最前：它是峰值最低的策略，先把最有价值的数字拿到手。
    if args.with_recursive:
        plans.append(("recursive", 0))
    if args.with_none:
        # 基准：完全不 checkpoint。注意它未必跑得起来（xl ctx2048 需要 32 × 6.6 GiB）
        plans.append(("none", n_layers))
    for k in args.block_sizes:
        plans.append(("block", k))

    print(f"{'策略':>10} {'block':>6} {'峰值(GiB)':>10} {'单步(ms)':>10}")
    def scrub():
        """清干净上一轮的残留。

        关键：上一次跑到一半 OOM 时，backward 已经给一部分参数分配了 .grad，
        这些张量（xl ≈ 6 GiB）如果不显式释放，**会让后面每一个策略都连锁 OOM**。
        一开始正是踩了这个坑，导致"探针单跑 recursive 成功、扫一遍却全 OOM"。
        """
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()

    for strategy, k in plans:
        args.strategy, args.block_size = strategy, k
        scrub()
        try:
            peak, mean_ms, std_ms = measure(model, x, y, args, warmup=args.warmup, steps=args.steps)
        except torch.cuda.OutOfMemoryError:
            scrub()
            rows.append((f"{strategy}/{k}", float("nan"), float("nan"), float("nan")))
            print(f"{strategy:>10} {k:>6} {'OOM':>10} {'—':>10}", flush=True)
            continue
        rows.append((f"{strategy}/{k}", peak, mean_ms, std_ms))
        print(f"{strategy:>10} {k:>6} {peak:>10.2f} {mean_ms:>10.1f}", flush=True)

    base = next((p for n, p, _, _ in rows if n.startswith("none") and p == p), float("nan"))
    if base == base:
        print()
        print(f"基准（不 checkpoint，全部留存）: {base:.2f} GiB")
        for name, peak, mean_ms, _ in rows:
            if peak != peak:
                continue
            print(f"  {name:>14}: 峰值 {peak:6.2f} GiB ({peak / base * 100:5.1f}% of 全留存), {mean_ms:7.1f} ms")
    del model, x, y
    gc.collect()
    torch.cuda.empty_cache()
    return 0


# --------------------------------------------------------------------------------------
# 子命令：verify-nesting —— 实测"外层 checkpoint 会抑制内层保存"这条机制
# --------------------------------------------------------------------------------------
def cmd_verify_nesting(args) -> int:
    """在 forward 阶段数一数到底留存了几个 checkpoint 输入张量。

    做法：把 N 层切成 N/k 段，每段再递归二分。若外层抑制生效，forward 阶段
    留存的张量数应 ≈ 递归深度（+ 常数），而不是 段数 × 每段内部节点数。
    """
    model, x, _ = build(args)
    blocks = list(model.layers)
    n = len(blocks)
    k = max(1, args.block_size)

    n_saved = 0
    n_bytes = 0

    def pack(t):
        nonlocal n_saved, n_bytes
        if isinstance(t, torch.nn.Parameter):
            return t
        n_saved += 1
        n_bytes += t.numel() * t.element_size()
        return t

    def unpack(t):
        return t

    def seg_rec(seg: list, h: torch.Tensor) -> torch.Tensor:
        if len(seg) == 1:
            return seg[0](h)
        mid = len(seg) // 2
        left, right = seg[:mid], seg[mid:]
        h = checkpoint(lambda inp, bl=left: seg_rec(bl, inp), h, use_reentrant=False)
        h = checkpoint(lambda inp, bl=right: seg_rec(bl, inp), h, use_reentrant=False)
        return h

    h = model.token_embeddings(x).detach().requires_grad_(True)
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        for i in range(0, n, k):
            seg = blocks[i : i + k]
            h = checkpoint(lambda inp, s=seg: seg_rec(s, inp), h, use_reentrant=False)

    per_tensor = args.batch_size * args.context_length * model_dims(args)[0] * (
        4 if args.dtype == "fp32" else 2
    )
    print(f"N={n} 层，外层每 {k} 层一个 checkpoint，段内二分递归")
    print(f"forward 阶段留存的残差张量: {n_saved} 个, 共 {n_bytes / MiB:.2f} MiB")
    print(f"  （一个 residual-stream 张量 = {per_tensor / MiB:.1f} MiB）")
    print(f"  对照：不嵌套的纯分段 需要留存 {n // k} 个 checkpoint 输入")
    print(f"        完全不留存（全 checkout）需要 {n} 层的全部残差")
    del model, x, h
    gc.collect()
    torch.cuda.empty_cache()
    return 0


# --------------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--model-size", choices=list(MODEL_SIZES), default="xl")
        sp.add_argument("--num-heads", type=int, default=None,
                        help="覆盖 Table 1 的 num_heads（讲义 §3.2 的 xl 片段用的是 16，Table 1 是 32）")
        sp.add_argument("--context-length", type=int, default=2048)
        sp.add_argument("--batch-size", type=int, default=4)
        sp.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
        sp.add_argument("--compile-blocks", action="store_true",
                        help="对每个 block 做 torch.compile(fullgraph=True) 融合（讲义 §3.2 的做法）")
        sp.add_argument("--device", default="cuda")
        sp.add_argument("--strategy", choices=["none", "block", "recursive"], default="none")
        sp.add_argument("--block-size", type=int, default=1)

    sp = sub.add_parser("residual", help="单个 block 为反向保存了多少残差（复现讲义 hook 实验）")
    common(sp)
    sp.add_argument("--compile", action="store_true",
                    help="torch.compile(block, fullgraph=True)，复现讲义 §3.2 的融合效果")
    sp.set_defaults(func=cmd_residual)

    sp = sub.add_parser("scan", help="扫描策略 / block size 的峰值显存与单步耗时")
    common(sp)
    sp.add_argument("--block-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    sp.add_argument("--with-recursive", action="store_true", help="额外测二分递归嵌套策略")
    sp.add_argument("--with-none", action="store_true",
                    help="额外测「完全不 checkpoint」的基准（大模型大概率 OOM，属预期）")
    sp.add_argument("--mode", choices=["forward", "full"], default="full",
                    help="forward=只前向（不含 backward），full=前向+反向")
    sp.add_argument("--warmup", type=int, default=2)
    sp.add_argument("--steps", type=int, default=5)
    sp.set_defaults(func=cmd_scan)

    sp = sub.add_parser("verify-nesting", help="实测外层 checkpoint 是否抑制内层保存")
    common(sp)
    sp.set_defaults(func=cmd_verify_nesting)

    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.device.startswith("cuda"):
        print("需要 CUDA 设备", file=sys.stderr)
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
