"""单个 TransformerBlock 的显存开销：**层数斜率法**（讲义 assignment 2 §2.2.6 (f)）。

为什么不用栈回溯直接归因到"第几层"：`TransformerBlock` 的 32 个实例跑的是**同一份代码**，
快照 `frames` 里的行号完全相同，无法区分第 0 层和第 31 层。

改用控制变量法——固定其它一切，只改 `num_layers`，测峰值，**斜率就是单层的边际开销**。
对均匀堆叠的 Transformer 这是精确的（实测两次斜率在线性上吻合到小数第四位）。

同时测两种模式，两者斜率之差就是**反向在单层上净增的显存**：
    forward-only 斜率  = 单层参数 + 该层为反向保存的残差
    full-step 斜率     = 上式 + 参数梯度 + 反向途中的临时激活梯度
（AdamW 的 exp_avg/exp_avg_sq 在 warmup 的第一次 step 就分配好了，
  且在测量前 reset_peak_memory_stats，所以**不落在斜率里**。）

用法：
    uv run python -m cs336_systems.memory_per_block \
        --d-model 768 --num-heads 12 --d-ff 3072 --context-length 512 --batch-size 4 \
        --layer-counts 6 9 12
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# cs336-basics 是本地子模块目录，不在 site-packages 里
_BASICS = Path(__file__).resolve().parent.parent / "cs336-basics"
if str(_BASICS) not in sys.path:
    sys.path.insert(0, str(_BASICS))

from cs336_basics.model import BasicsTransformerLM  # noqa: E402

MiB = 1024 * 1024
GiB = 1024 * 1024 * 1024


def peak_gib(
    num_layers: int,
    *,
    d_model: int,
    num_heads: int,
    d_ff: int,
    context_length: int,
    batch_size: int,
    vocab_size: int,
    full_step: bool,
    dtype: torch.dtype,
    warmup: int,
    device: str,
) -> float:
    """建模型 → warmup（让 allocator/优化器状态稳定）→ reset_peak → 测一步峰值。"""
    torch.manual_seed(0)
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
    ).to(device)
    if dtype != torch.float32:
        model = model.to(dtype)

    x = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    y = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def one_step() -> None:
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        if full_step:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    # warmup：第一次 step 会把 AdamW 状态分配出来，必须发生在 reset 之前
    for _ in range(warmup):
        one_step()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    one_step()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / GiB

    del model, optimizer, x, y
    gc.collect()
    torch.cuda.empty_cache()
    return peak


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--d-model", type=int, default=768)
    p.add_argument("--num-heads", type=int, default=12)
    p.add_argument("--d-ff", type=int, default=3072)
    p.add_argument("--context-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--vocab-size", type=int, default=10000)
    p.add_argument("--layer-counts", type=int, nargs="+", default=[6, 9, 12])
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--device", default="cuda")
    p.add_argument("--forward-only", action="store_true", help="只测 forward，不测 full step")
    args = p.parse_args()

    if not args.device.startswith("cuda"):
        print("需要 CUDA 设备来测显存", file=sys.stderr)
        return 2
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16

    print(
        f"配置: d_model={args.d_model} heads={args.num_heads} d_ff={args.d_ff} "
        f"ctx={args.context_length} batch={args.batch_size} dtype={args.dtype} "
        f"warmup={args.warmup}"
    )

    n_layers_total = args.layer_counts[-1]
    modes = [("forward-only", False)] + ([] if args.forward_only else [("full-step", True)])

    peaks: dict[str, dict[int, float]] = {}
    for mode_name, full_step in modes:
        peaks[mode_name] = {}
        for n in args.layer_counts:
            gib = peak_gib(
                n,
                d_model=args.d_model,
                num_heads=args.num_heads,
                d_ff=args.d_ff,
                context_length=args.context_length,
                batch_size=args.batch_size,
                vocab_size=args.vocab_size,
                full_step=full_step,
                dtype=dtype,
                warmup=args.warmup,
                device=args.device,
            )
            peaks[mode_name][n] = gib
            print(f"  [{mode_name:12s}] num_layers={n:3d}  peak={gib:7.3f} GiB")

    ns = sorted(args.layer_counts)
    slopes: dict[str, tuple[list[float], float]] = {}
    print()
    for mode_name in peaks:
        per_pair = [
            (peaks[mode_name][b] - peaks[mode_name][a]) / (b - a) for a, b in zip(ns, ns[1:])
        ]
        avg = sum(per_pair) / len(per_pair)
        slopes[mode_name] = (per_pair, avg)
        spread = max(per_pair) - min(per_pair)
        print(
            f"[{mode_name:12s}] 每层增量: "
            + ", ".join(f"{v * 1024:.4f} MiB" for v in per_pair)
            + f"  → 平均 {avg * 1024:.1f} MiB/block（组间极差 {spread * 1024:.4f} MiB）"
        )

    if len(slopes) == 2:
        fwd = slopes["forward-only"][1] * 1024
        full = slopes["full-step"][1] * 1024
        print()
        print(f"full-step − forward-only = {full - fwd:.1f} MiB/block（反向净增）")
        # 解析对账：该层参数（含两个 RMSNorm）
        per_layer_params = 4 * args.d_model**2 + 3 * args.d_model * args.d_ff + 2 * args.d_model
        bytes_per_param = 4 if args.dtype == "fp32" else 2
        grad_mib = per_layer_params * bytes_per_param / MiB
        print(f"该层参数量 = {per_layer_params / 1e6:.4f} M → 参数梯度(同尺寸) = {grad_mib:.2f} MiB")
        print(f"forward-only 斜率 − 该层参数 = {fwd - grad_mib:.1f} MiB/block（纯残差，fp32 参数）")
        print(f"其中临时激活梯度 ≈ {full - fwd - grad_mib:.1f} MiB/block")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
