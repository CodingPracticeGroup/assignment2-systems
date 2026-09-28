"""CS336 Assignment 2 -- Problem (nsys_profile) 的自动化驱动。

讲义要求：选 2 个模型规模 × 3 个 2 的幂的 context 长度（>128，最大者取「显存能装下的最长」），
分别 profile forward / backward / optimizer step，然后回答 (a)-(d)。

这个脚本把整件事自动化：

    probe    找出每个 model size 能装下的最大 context length
    profile  对每个 (size, ctx) 跑 nsys profile（forward 与 full 各一份）
    report   导出报表并汇总成 markdown（含 (a)-(d) 需要的数字）

用法：
    uv run python -m cs336_systems.profile_nsys probe   --model-sizes small medium
    uv run python -m cs336_systems.profile_nsys profile --model-sizes small medium \\
        --context-lengths 256 512 1024 --outdir profiles
    uv run python -m cs336_systems.profile_nsys report  --outdir profiles

⚠️ 本机（nsys 2025.3.2 + torch 2.11）实测的两个坑，脚本已绕开：
  1. `-c nvtx -p <range>` 触发不了采集 → 改用 `-c cudaProfilerApi`（benchmark 的 --profiler-range）
  2. PyTorch autograd 在独立线程发射 kernel，NVTX 区间盖不住 backward
     → forward 靠 NVTX 归因，backward/optimizer 靠「forward-only vs full 两份 profile 的差值」
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NSYS = shutil.which("nsys") or "/usr/local/cuda-13.0/bin/nsys"

# cuBLAS 的 matmul kernel 命名（Ada/Ampere 世代都叫 ampere_sgemm 或 cutlass_*_gemm）
_MATMUL_RE = re.compile(r"sgemm|hgemm|bgemm|cutlass.*gemm", re.IGNORECASE)

# 讲义 §2.1.2 的规模表里，16GB 卡上现实可行的两个
DEFAULT_SIZES = ["small", "medium"]
DEFAULT_CONTEXTS = [128, 256, 512, 1024, 2048]   # probe 用；最终取 3 个


def run(cmd: list[str], timeout: int | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=timeout)


# --------------------------------------------------------------------------- #
# 1) probe：找能装下的最大 context length
# --------------------------------------------------------------------------- #
def bench_once(size: str, ctx: int, mode: str, warmup: int, steps: int,
               profiler_range: bool = False, nvtx: bool = False) -> tuple[bool, str]:
    """跑一次 benchmark，成功则返回 (True, RESULT 行)。"""
    cmd = [sys.executable, "-m", "cs336_systems.benchmark",
           "--model-size", size, "--context-length", str(ctx), "--mode", mode,
           "--warmup", str(warmup), "--steps", str(steps)]
    if profiler_range:
        cmd.append("--profiler-range")
    if nvtx:
        cmd.append("--nvtx")
    p = run(cmd)
    line = next((ln for ln in p.stdout.splitlines() if ln.startswith("RESULT")), "")
    return bool(line), line


def cmd_probe(args: argparse.Namespace) -> int:
    print(f"probe：mode={args.mode}，候选 context = {args.context_lengths}\n")
    results: dict[str, int] = {}
    for size in args.model_sizes:
        best = 0
        for ctx in sorted(args.context_lengths):
            ok, line = bench_once(size, ctx, args.mode, warmup=1, steps=1)
            flag = "OK " if ok else "OOM"
            extra = ""
            if ok:
                peak = re.search(r"peak_gb=([\d.]+)", line)
                extra = f"峰值 {peak.group(1)} GB" if peak else ""
            print(f"  {size:7s} ctx={ctx:5d}  {flag}  {extra}")
            if ok:
                best = ctx
            elif ctx > best:          # 显存随 ctx 单调增长，第一次失败即可停
                break
        results[size] = best
        print(f"  → {size}: 最大可装 context = {best}\n")
    return 0


# --------------------------------------------------------------------------- #
# 2) profile：跑 nsys
# --------------------------------------------------------------------------- #
def cmd_profile(args: argparse.Namespace) -> int:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    combos = [(s, c) for s in args.model_sizes for c in args.context_lengths]
    print(f"将产出 {len(combos) * 2} 份 profile 到 {outdir}\n")

    for size, ctx in combos:
        for mode in ("forward", "full"):
            tag = f"{size}_ctx{ctx}_{mode}"
            rep = outdir / f"{tag}.nsys-rep"
            if args.skip_existing and rep.exists():
                print(f"  跳过 {tag}（已存在）")
                continue

            # nsys 侧：用 cudaProfilerApi 触发，跳过 warmup
            cmd = [NSYS, "profile", "--force-overwrite=true", "-o", str(outdir / tag),
                   "--trace=cuda,nvtx", "-c", "cudaProfilerApi", "--capture-range-end=stop",
                   "--", sys.executable, "-m", "cs336_systems.benchmark",
                   "--model-size", size, "--context-length", str(ctx), "--mode", mode,
                   "--warmup", str(args.warmup), "--steps", str(args.steps),
                   "--nvtx", "--profiler-range"]
            p = run(cmd, timeout=args.timeout)
            size_kb = rep.stat().st_size // 1024 if rep.exists() else 0
            status = "OK " if size_kb else "FAIL"
            print(f"  {status} {tag:28s} {size_kb:6d} KB")
            if not size_kb:
                print("        stderr:", (p.stderr or "").strip().splitlines()[-1:] or "")
    return 0


# --------------------------------------------------------------------------- #
# 3) report：导出报表 + 汇总
# --------------------------------------------------------------------------- #
def stats_csv(rep: Path, report: str) -> list[list[str]]:
    p = run([NSYS, "stats", "--force-export=true", "--report", report,
             "--format", "csv", str(rep)], timeout=600)
    return [r for r in csv.reader(io.StringIO(p.stdout)) if len(r) >= 9]


def kernel_summary(rep: Path) -> tuple[float, list[tuple[float, int, str]]]:
    """返回 (matmul 时间占比 %, [(时间占比, 调用次数, kernel名), ...] 按耗时降序)。"""
    rows = stats_csv(rep, "cuda_gpu_kern_sum")
    mm = other = 0.0
    kept: list[tuple[float, int, str]] = []
    for r in rows:
        try:
            pct, inst = float(r[0]), int(r[2])
        except ValueError:
            continue
        name = ",".join(r[8:]).strip()
        kept.append((pct, inst, name))
        if _MATMUL_RE.search(name):
            mm += pct
        else:
            other += pct
    total = mm + other
    return (100.0 * mm / total if total else 0.0), kept


def nvtx_projection(rep: Path) -> dict[str, float]:
    """把 GPU 时间投影到 NVTX 区间 → {区间名: 投影时间 ms}。"""
    out: dict[str, float] = {}
    for r in stats_csv(rep, "nvtx_gpu_proj_sum"):
        # 列：Range, Style, Total Proj Time (ns), Total Range Time (ns), ...
        try:
            out[r[0].strip()] = float(r[2]) / 1e6
        except (ValueError, IndexError):
            continue
    return out


def fmt_table(rows: list[tuple[float, int, str]], top: int = 8) -> str:
    out = ["| Time % | Instances | Kernel |", "|---:|---:|---|"]
    for pct, inst, name in rows[:top]:
        short = name if len(name) <= 70 else name[:67] + "..."
        out.append(f"| {pct:.1f}% | {inst} | `{short}` |")
    return "\n".join(out)


def cmd_baseline(args: argparse.Namespace) -> int:
    """不带 nsys 跑一遍（讲义 §2.1.3 的 timeit 数字），供 (a) 对比 nsys 的 forward 时间。"""
    import json
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data: dict[str, dict] = {}
    for size in args.model_sizes:
        for ctx in args.context_lengths:
            rec = {}
            for mode in ("forward", "full"):
                ok, line = bench_once(size, ctx, mode, args.warmup, args.steps)
                rec[mode] = line if ok else None
                ms = re.search(r"mean_ms=([\d.]+)", line) if ok else None
                print(f"  {size:7s} ctx={ctx:5d} {mode:16s} {'OOM' if not ok else ms.group(1) + ' ms'}")
            data[f"{size}_ctx{ctx}"] = rec
    dest = outdir / "baseline.json"
    dest.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已写出 {dest}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    import json
    outdir = Path(args.outdir)
    reps = sorted(outdir.glob("*.nsys-rep"))
    if not reps:
        print(f"{outdir} 下没有 .nsys-rep，先跑 profile")
        return 2

    base: dict = {}
    bfile = outdir / "baseline.json"
    if bfile.exists():
        base = json.loads(bfile.read_text(encoding="utf-8"))

    # ---- 汇总：按 (size, ctx) 归并 forward / full ----
    summary: dict[tuple[str, str], dict] = {}
    details: list[tuple[str, float, list]] = []
    for rep in reps:
        tag = rep.stem
        m = re.match(r"(.+)_ctx(\d+)_(forward|full)$", tag)
        if not m:
            continue
        size, ctx, mode = m.group(1), m.group(2), m.group(3)
        mm_pct, kept = kernel_summary(rep)
        proj = nvtx_projection(rep)
        summary.setdefault((size, ctx), {})[mode] = {"mm": mm_pct, "proj": proj}
        details.append((tag, mm_pct, kept))

    lines = ["# nsys_profile 结果汇总（自动生成）", ""]

    # ---- (d) 的核心对比表 ----
    lines += ["## ① forward-only vs 完整训练步：matmul 占比（讲义 (d)）", "",
              "| 模型 | context | forward matmul% | full matmul% | 变化 | nsys measure/步 fwd (ms) | timeit fwd (ms) | nsys measure/步 full (ms) | timeit full (ms) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    n = max(1, args.steps)
    for (size, ctx), d in sorted(summary.items()):
        f, u = d.get("forward", {}), d.get("full", {})
        if not f or not u:
            continue
        b = base.get(f"{size}_ctx{ctx}", {})

        def g(rec, key):
            s = rec.get(key) if rec else None
            m = re.search(r"mean_ms=([\d.]+)", s) if s else None
            return f"{float(m.group(1)):.1f}" if m else "—"

        # 注意：nsys 的 :measure 覆盖全部计时步，要除以步数才能和 timeit 的单步均值比
        lines.append(
            f"| {size} | {ctx} | {f['mm']:.1f}% | {u['mm']:.1f}% | "
            f"{u['mm'] - f['mm']:+.1f} pt | {f['proj'].get(':measure', float('nan')) / n:.1f} | "
            f"{g(b,'forward')} | {u['proj'].get(':measure', float('nan')) / n:.1f} | {g(b,'full')} |")
    lines += ["",
              f"> `nsys measure/步` = `nvtx_gpu_proj_sum` 里 `:measure` 的投影 GPU 时间 ÷ {n} 步；",
              "> 和 timeit 的单步均值应当接近（若差得远，说明 profiling 本身的开销不可忽略）。", ""]

    # ---- 每份 profile 的明细 ----
    lines += ["## ② 逐份 profile 明细", ""]
    for tag, mm_pct, kept in details:
        lines += [f"### `{tag}`", "",
                  f"- **matmul 时间占比：{mm_pct:.1f}%**（其余 {100 - mm_pct:.1f}%）", ""]
        lines += [fmt_table(kept), ""]

    lines += [
        "---", "",
        "## ⚠️ NVTX 投影的两个已知陷阱（本机实测）", "",
        "1. **`:backward` 的投影≈0，不可信。** PyTorch 的 autograd 引擎在**独立的 device 线程**上",
        "   发射 kernel，主线程上 `torch.cuda.nvtx.range(\"backward\")` 推的区间盖不住它们",
        "   （实测：backward 只归因到 1 个 kernel，而 34 种 kernel / 936ms 完全未被任何区间覆盖）。",
        "   要覆盖 backward 得加 `--pytorch=autograd-nvtx`，但那样区间会细到**每个算子一个**",
        "   （`aten::_foreach_addcdiv_, op_id=18667`），反而不好聚合。",
        "2. **`:optimizer` 的投影被高估。** `optimizer.step()` 在 CPU 上紧接着 `backward()` 返回就执行，",
        "   而 backward 的 GPU kernel 还在飞——于是**反向的尾巴被算进了 optimizer 的窗口**",
        "   （实测 medium_ctx128 的 `:optimizer` 显示 395ms，比整个 forward 的 147ms 还大，不可能）。", "",
        "**结论：只有两个数字可信** ——",
        "`:measure`（整步的 GPU 时间）与 `:forward`（每步第一段，前面没有异步尾巴）。",
        "**backward + optimizer 的量请用「forward-only 与 full 两份 profile 的差值」得到**，" ,
        "这正是上面 ① 表的用法。",
    ]

    dest = outdir / "REPORT.md"
    dest.write_text("\n".join(lines), encoding="utf-8")
    print(f"已写出 {dest}（{len(reps)} 份 profile）")
    return 0


# --------------------------------------------------------------------------- #
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("probe", help="找每个规模能装下的最大 context length")
    q.add_argument("--model-sizes", nargs="+", default=DEFAULT_SIZES)
    q.add_argument("--context-lengths", nargs="+", type=int, default=DEFAULT_CONTEXTS)
    q.add_argument("--mode", default="full")
    q.set_defaults(func=cmd_probe)

    r = sub.add_parser("profile", help="跑 nsys profile")
    r.add_argument("--model-sizes", nargs="+", default=DEFAULT_SIZES)
    r.add_argument("--context-lengths", nargs="+", type=int, required=True)
    r.add_argument("--outdir", default="profiles")
    r.add_argument("--warmup", type=int, default=2)
    r.add_argument("--steps", type=int, default=3)
    r.add_argument("--timeout", type=int, default=900)
    r.add_argument("--skip-existing", action="store_true")
    r.set_defaults(func=cmd_profile)

    s = sub.add_parser("report", help="导出报表并汇总成 markdown")
    s.add_argument("--outdir", default="profiles")
    s.add_argument("--steps", type=int, default=3,
                   help="profile 时用的计时步数（用于把 nsys 的总时间折算成单步）")
    s.set_defaults(func=cmd_report)

    b = sub.add_parser("baseline", help="不带 nsys 跑一遍，存 timeit 数字（供 (a) 对比）")
    b.add_argument("--model-sizes", nargs="+", default=DEFAULT_SIZES)
    b.add_argument("--context-lengths", nargs="+", type=int, required=True)
    b.add_argument("--outdir", default="profiles")
    b.add_argument("--warmup", type=int, default=5)
    b.add_argument("--steps", type=int, default=10)
    b.set_defaults(func=cmd_baseline)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
