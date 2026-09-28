"""CS336 Assignment 2 (systems) -- end-to-end benchmarking.

问题 (benchmarking_script) 要求的脚本：建模型 -> 造随机数据 -> warmup w 步 ->
计时 n 步（forward / forward+backward / 完整训练步）。

设计原则（讲义 §2.1.3 明确建议）：
    后面会反复改精度、换层，所以所有变化点都通过命令行参数暴露出来。

用法示例（从 repo 根目录运行）：
    uv run python -m cs336_systems.benchmark --model-size small --mode forward
    uv run python -m cs336_systems.benchmark --model-size medium --mode full --dtype bf16

    # 对能装下的规模扫一遍（large/xl/10B 在 16GB 卡上会 OOM）：
    for s in small medium; do
      for m in forward forward-backward full; do
        uv run python -m cs336_systems.benchmark --model-size $s --mode $m
      done
    done

配合 nsys（讲义 §2.1.4）：
    nsys profile --force-overwrite=true -o /tmp/prof -- \
      uv run python -m cs336_systems.benchmark --model-size small --mode full
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import timeit

import torch

from cs336_basics.model import BasicsTransformerLM

# 讲义 §2.1.2 的模型规格表（除 leaderboard 外，vocab=10000、batch=4、context=512）
MODEL_SIZES: dict[str, dict[str, int]] = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    "10B": {"d_model": 4608, "d_ff": 12288, "num_layers": 50, "num_heads": 36},
}

VOCAB_SIZE = 10_000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-size", choices=sorted(MODEL_SIZES), default="small")
    p.add_argument("--mode", choices=["forward", "forward-backward", "full"], default="forward",
                   help="full = forward + backward + optimizer step")
    p.add_argument("--context-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    p.add_argument("--warmup", type=int, default=5, help="不计时的预热步数")
    p.add_argument("--steps", type=int, default=10, help="计时的步数")
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32",
                   help="bf16 走 autocast 混合精度（下一题 benchmarking_mixed_precision 会用到）")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--nvtx", action="store_true",
                   help="打开 NVTX 埋点（供 nsys 报表做 forward/backward/optimizer 归因）")
    p.add_argument("--profiler-range", action="store_true",
                   help="用 cudaProfilerStart/Stop 把正式计时段包起来；"
                        "配合 nsys -c cudaProfilerApi 跳过 warmup（本机实测：-c nvtx 触发不灵，此路可用）")
    return p.parse_args()


def nvtx(msg: str, enabled: bool):
    """NVTX 区间；未开启时是零成本 no-op。

    nsys 用它来：(1) 圈出正式计时段以跳过 warmup；(2) 把 GPU 时间归因到
    forward / backward / optimizer（`nsys stats --report nvtx_gpu_proj_sum`）。
    """
    if not enabled:
        return contextlib.nullcontext()
    return torch.cuda.nvtx.range(msg)


def make_model(args: argparse.Namespace) -> BasicsTransformerLM:
    cfg = MODEL_SIZES[args.model_size]
    return BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
    ).to(args.device)


def make_batch(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor]:
    """随机 token 与随机 target（只测速度/显存，不需要真实数据）。"""
    x = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=args.device)
    y = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=args.device)
    return x, y


def make_amp_context(args: argparse.Namespace):
    """fp32 -> no-op 上下文；bf16 -> autocast（讲义提示可以用 nullcontext）。"""
    if args.dtype == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def run_step(model, optimizer, x, y, args) -> None:
    """跑一步，按 --mode 决定做什么。

    NVTX 区间按「forward / backward / optimizer」分开，这样 nsys 的
    nvtx_gpu_proj_sum 报表能直接把 GPU 时间拆到这三段上（对应讲义 (a)(d)）。
    """
    with nvtx("forward", args.nvtx):
        with make_amp_context(args):
            logits = model(x)
            if args.mode == "forward":
                return
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1)
            )
    with nvtx("backward", args.nvtx):
        loss.backward()
    if args.mode == "full":
        with nvtx("optimizer", args.nvtx):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.device.startswith("cuda"):
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()

    try:
        model = make_model(args)
    except torch.OutOfMemoryError:
        print(f"OOM   model={args.model_size} (建模型就放不下)")
        return 2

    # 只有 full 模式才需要优化器（省下 Adam 的 2x 参数显存，见笔记里的显存分析）
    optimizer = None
    if args.mode == "full":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x, y = make_batch(args)

    # ---- warmup（不计时）----
    # 讲义 (c) 就是问：不做 warmup / 只做 1-2 步会怎样
    # 用 NVTX 把 warmup 圈起来，nsys 侧就能用 -c nvtx -p measure 只采计时段
    with nvtx("warmup", args.nvtx):
        for _ in range(args.warmup):
            run_step(model, optimizer, x, y, args)

    # ---- 计时 ----
    # 只采这一段：nsys -c cudaProfilerApi --capture-range-end=stop
    if args.profiler_range:
        torch.cuda.profiler.start()
    times: list[float] = []
    with nvtx("measure", args.nvtx):
        for _ in range(args.steps):
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()          # 计时前：等前面所有 kernel 排空
            t0 = timeit.default_timer()
            run_step(model, optimizer, x, y, args)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()          # 计时后：等这一步真的跑完
            times.append(timeit.default_timer() - t0)
    if args.profiler_range:
        torch.cuda.profiler.stop()

    t = torch.tensor(times, dtype=torch.float64)
    mean_ms, std_ms = t.mean().item() * 1e3, t.std(unbiased=True).item() * 1e3
    peak_gb = 0.0
    if args.device.startswith("cuda"):
        peak_gb = torch.cuda.max_memory_allocated() / 2**30

    # 一行一条，方便后面拼成表格
    print(
        f"RESULT model={args.model_size} mode={args.mode} dtype={args.dtype} "
        f"warmup={args.warmup} steps={args.steps} "
        f"mean_ms={mean_ms:.3f} std_ms={std_ms:.3f} peak_gb={peak_gb:.2f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
