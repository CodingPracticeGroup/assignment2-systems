"""讲义 Problem (benchmarking_mixed_precision) (a)(b) 的实验部分。

(a) 用 forward hook + backward 后检查，**实测**下面这些部件的 dtype：
    模型参数 / fc1 输出 / LayerNorm 输出 / logits / loss / 梯度
    分别在「autocast(fp16) 内」与「autocast(fp16) 外」两种情况下测。

(b) 再看 LayerNorm 内部各步（mean/var 的归约、归一化、仿射）对精度的敏感度，
    并对比 fp16 与 bf16 的动态范围。

运行：uv run python -m cs336_systems.mixed_precision_dtypes
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn as nn

NAME = lambda d: str(d).replace("torch.", "")  # noqa: E731


class ToyModel(nn.Module):
    """讲义原文的 ToyModel（一字未改）。"""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x


def probe(amp_dtype: torch.dtype | None) -> dict[str, str]:
    """跑一次 forward+backward，返回各部件实测到的 dtype。"""
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = ToyModel(16, 8).to(dev)

    seen: dict[str, str] = {}
    hooks = []
    for name, mod in (("fc1", model.fc1), ("ln", model.ln), ("fc2", model.fc2)):
        def make(n):
            def hook(_m, _inp, out):
                seen[n] = NAME(out.dtype)
            return hook
        hooks.append(mod.register_forward_hook(make(name)))

    # 参数 dtype（在 autocast 上下文**内部**读，看有没有被改）
    ctx = (
        torch.autocast(device_type=dev, dtype=amp_dtype) if amp_dtype else contextlib.nullcontext()
    )
    with ctx:
        seen["param (fc1.weight) inside autocast"] = NAME(model.fc1.weight.dtype)
        x = torch.randn(4, 16, device=dev)
        logits = model(x)
        seen["logits (fc2 out)"] = NAME(logits.dtype)
        target = torch.randint(0, 8, (4,), device=dev)
        loss = nn.functional.cross_entropy(logits, target)
        seen["loss"] = NAME(loss.dtype)
    loss.backward()
    seen["grad (fc1.weight.grad)"] = NAME(model.fc1.weight.grad.dtype)
    seen["param after backward"] = NAME(model.fc1.weight.dtype)
    for h in hooks:
        h.remove()
    return seen


def dtype_range(d: torch.dtype) -> str:
    fi = torch.finfo(d)
    return f"min±{fi.min:.3e}  max {fi.max:.3e}  最小正规数 {fi.tiny:.3e}  eps {fi.eps:.3e}"


def ln_precision_demo() -> None:
    """LayerNorm 内部：归约量 vs 归一化结果的量级差异（解释为什么归约要 fp32）。"""
    torch.manual_seed(0)
    x = torch.randn(4, 10, device="cuda" if torch.cuda.is_available() else "cpu") * 3 + 5

    mean32, var32 = x.mean(-1), x.var(-1, unbiased=False)
    mean16, var16 = x.half().mean(-1).float(), x.half().var(-1, unbiased=False).float()
    print("\n--- LayerNorm 的归约量：fp16 重算 vs fp32 ---")
    print(f"  fp32 mean/var : {mean32[0].item():+.8f} / {var32[0].item():+.8f}")
    print(f"  fp16 mean/var : {mean16[0].item():+.8f} / {var16[0].item():+.8f}")
    print(f"  → mean 绝对误差 {abs(mean16[0]-mean32[0]).item():.3e}，"
          f"var 绝对误差 {abs(var16[0]-var32[0]).item():.3e}")

    # 归一化后要除以 sqrt(var + eps)：var 上有微小误差会被放大
    eps = 1e-5
    out32 = (x - mean32[:, None]) / torch.sqrt(var32[:, None] + eps)
    out16 = (x.half() - mean16[:, None].half()) / torch.sqrt(var16[:, None].half() + eps)
    print(f"  归一化输出的最大差: {abs(out32 - out16.float()).max().item():.3e}")


def main() -> int:
    print("=" * 78)
    print("(a) 各部件 dtype 实测")
    print("=" * 78)
    for amp, label in (
        (None, "autocast 关闭（全 fp32）"),
        (torch.float16, "autocast(fp16)"),
        (torch.bfloat16, "autocast(bf16)"),
    ):
        print(f"\n### {label}")
        for k, v in probe(amp).items():
            print(f"  {k:36s} → {v}")

    print("\n" + "=" * 78)
    print("(b) 为什么 LayerNorm 特殊：fp16 vs bf16 的动态范围")
    print("=" * 78)
    print(f"  fp16: {dtype_range(torch.float16)}")
    print(f"  bf16: {dtype_range(torch.bfloat16)}")
    print(f"  fp32: {dtype_range(torch.float32)}")
    ln_precision_demo()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
