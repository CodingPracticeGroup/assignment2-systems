"""讲义 Problem (mixed_precision_accumulation)：1 点。

四段几乎相同的循环，唯一的区别在 **累加器（s）的精度** 与 **加数（0.01）的精度**：

    ① s fp32, 加数 fp32          —— 全精度基准
    ② s fp16, 加数 fp16          —— 累加也在 fp16 里做（灾难）
    ③ s fp32, 加数 fp16          —— 累加器 fp32，加数被自动提升（type promotion）
    ④ s fp32, 加数 fp16 显式 .type(float32) —— 和 ③ 等价？

运行：uv run python -m cs336_systems.mixed_precision_accumulation
"""

from __future__ import annotations

from decimal import Decimal

import torch

N = 1000
STEP = 0.01


def f32(x: torch.Tensor) -> Decimal:
    """把张量里的 32-bit 浮点值转成精确十进制，便于看真实误差。"""
    return Decimal(float(x))


def run_all() -> list[tuple[str, float, Decimal]]:
    out: list[tuple[str, float, Decimal]] = []

    # ① 全 fp32：累加器与加数都是 fp32
    s = torch.tensor(0, dtype=torch.float32)
    for _ in range(N):
        s += torch.tensor(STEP, dtype=torch.float32)
    out.append(("① s=fp32, 加数=fp32", float(s), f32(s)))

    # ② 全 fp16：**累加器也是 fp16**，每一步都在 fp16 里舍入
    s = torch.tensor(0, dtype=torch.float16)
    for _ in range(N):
        s += torch.tensor(STEP, dtype=torch.float16)
    out.append(("② s=fp16, 加数=fp16", float(s), f32(s)))

    # ③ 累加器 fp32，加数 fp16（靠 PyTorch type promotion 提升到 fp32 再相加）
    s = torch.tensor(0, dtype=torch.float32)
    for _ in range(N):
        s += torch.tensor(STEP, dtype=torch.float16)
    out.append(("③ s=fp32, 加数=fp16", float(s), f32(s)))

    # ④ 同上，但加数显式 cast 到 fp32 后再加
    s = torch.tensor(0, dtype=torch.float32)
    for _ in range(N):
        x = torch.tensor(STEP, dtype=torch.float16)
        s += x.type(torch.float32)
    out.append(("④ s=fp32, 加数 fp16→fp32", float(s), f32(s)))

    return out


def main() -> int:
    exact = Decimal(N) * Decimal("0.01")          # 数学上的精确值 = 10

    print(f"循环 {N} 次，每次加 0.01；数学精确值 = {exact}\n")

    print(f"{'方案':28s} {'结果(float)':>14s} {'误差':>12s} {'相对误差':>12s}")
    print("-" * 70)
    for name, val, dec in run_all():
        err = dec - exact
        rel = err / exact
        print(f"{name:28s} {val:14.6f} {float(err):+12.6f} {float(rel):+12.3e}")

    print("\n--- 相关的表示误差（三者不是一回事，务必分清）---")
    step32 = torch.tensor(STEP, dtype=torch.float32)
    step16 = torch.tensor(STEP, dtype=torch.float16)
    print(f"0.01 在 fp32 里的值 : {float(step32):.12f}   误差 {float(step32) - STEP:+.3e}")
    print(f"0.01 在 fp16 里的值 : {float(step16):.12f}   误差 {float(step16) - STEP:+.3e}")
    print(f"   → 1000 次累加后，只由加数表示误差造成的偏差 = "
          f"{float(step16) * N - N * STEP:+.6f}")
    print(f"fp16 在 10 附近的 ULP: {2.0 ** (3 - 10):.8f}"
          f"   （比 0.01 小一点，所以不会完全加不上去，但每步都被舍入）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
