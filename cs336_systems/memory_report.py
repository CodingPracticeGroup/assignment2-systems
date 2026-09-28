"""解析 torch.cuda.memory._dump_snapshot 产出的 pickle。

为什么不用 pytorch.org/memory_viz：那是个网页工具，**图不便版本化、也不便自动化**。
这里用 snapshot 里现成的 `device_traces`（每条含 `time_us` / `size` / `frames`）
自己重建同一条 **Active Memory Timeline**，并存成 PNG；同时把**最大的若干次分配**
及其 **Python 栈回溯**打出来（对应讲义 memory_profiling (b)(e)）。

用法：
    uv run python -m cs336_systems.memory_report SNAPSHOT.pickle \
        --plot out.png --top 10 [--phase-frac 0.28 0.88 --title "..."]
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无显示环境
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

MiB = 1024 * 1024

# 让图里的中文标题能正常显示（DejaVu Sans 没有 CJK 字形）
for _cand in (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
):
    try:
        fm.fontManager.addfont(_cand)
        plt.rcParams["font.sans-serif"] = [fm.FontProperties(fname=_cand).get_name()]
        plt.rcParams["axes.unicode_minus"] = False
        break
    except Exception:  # noqa: BLE001 - 字体缺失时静默回退
        continue


def load(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def phase_markers(snap: dict, device: int = 0) -> list[tuple[str, float]]:
    """从快照的 external_annotations 里取出 PHASE_* 的 START 位置（快照时钟的分数）。

    这些区间由 benchmark 的 `record_function("PHASE_xxx")` 写入；
    用快照自己的时钟标阶段最可靠（墙钟比例会错位，见模块 docstring）。
    """
    evs = [e for e in snap["device_traces"][device] if e["action"] in ("alloc", "free_completed")]
    if not evs:
        return []
    t0, span = min(e["time_us"] for e in evs), max(e["time_us"] for e in evs) - min(
        e["time_us"] for e in evs
    )
    if span <= 0:
        return []
    out = []
    for a in snap.get("external_annotations") or []:
        if a.get("stage") == "START" and a.get("name", "").startswith("PHASE_"):
            out.append((a["name"].removeprefix("PHASE_"), (a["time_us"] - t0) / span))
    return sorted(out, key=lambda x: x[1])


def baseline_mib(snap: dict, device: int = 0) -> float:
    """录制开始**之前**就已存在的已分配内存（最主要的就是模型参数）。

    `_record_memory_history` 只记录开启之后的事件，所以模型参数的 alloc 不在
    `device_traces` 里，必须补上这个基线，否则时间线的 y 轴会低估一整个参数量。
    反推：dump 时的活跃量 = 基线 + Σalloc − Σfree_completed。
    """
    live = sum(
        b["size"]
        for seg in snap["segments"]
        if seg.get("device") == device
        for b in seg["blocks"]
        if b["state"] == "active_allocated"
    )
    evs = snap["device_traces"][device]
    a = sum(e["size"] for e in evs if e["action"] == "alloc")
    f = sum(e["size"] for e in evs if e["action"] == "free_completed")
    return (live - a + f) / MiB


def build_timeline(snap: dict, device: int = 0):
    """把 device_traces 的事件流重放成 (时间 ms, 当前已分配 MiB)，含录制前基线。"""
    evs = [e for e in snap["device_traces"][device] if e["action"] in ("alloc", "free_completed")]
    evs.sort(key=lambda e: e["time_us"])
    if not evs:
        return [], [], 0.0, 0.0
    base = baseline_mib(snap, device)
    t0 = evs[0]["time_us"]
    ts, ys, cur, peak, peak_t = [], [], 0, 0, 0.0
    for e in evs:
        cur += e["size"] if e["action"] == "alloc" else -e["size"]
        t = (e["time_us"] - t0) / 1000.0          # µs -> ms
        ts.append(t)
        ys.append(base + cur / MiB)
        if base + cur / MiB > peak:
            peak, peak_t = base + cur / MiB, t
    return ts, ys, peak, peak_t


def top_allocations(snap: dict, n: int, device: int = 0) -> list[tuple[int, list[dict]]]:
    """按大小取前 n 次 alloc，附带栈回溯。"""
    evs = [e for e in snap["device_traces"][device] if e["action"] == "alloc"]
    evs.sort(key=lambda e: -e["size"])
    return [(e["size"], e.get("frames") or []) for e in evs[:n]]


def py_frames(frames: list[dict], keep: int = 6) -> list[str]:
    """只保留 Python 帧（C++ 内部帧对定位来源没用），取最靠近调用点的那几层。"""
    out = [f"{f['name']}  ({Path(f['filename']).name}:{f['line']})"
           for f in frames
           if f.get("filename", "").endswith(".py")]
    return out[:keep] or ["  <无 Python 帧>"]


def plot(ts: list[float], ys: list[float], out: Path, title: str,
         phases: list[tuple[str, float]] | None = None, peak: float = 0.0, peak_t: float = 0.0,
         xlabel: str = "time (ms)") -> None:
    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=130)
    ax.plot(ts, ys, linewidth=0.9, color="#1f77b4")
    ax.fill_between(ts, ys, alpha=0.22, color="#1f77b4")

    if peak:
        ax.axhline(peak, ls="--", lw=0.9, color="crimson", alpha=0.8)
        ax.annotate(f"peak {peak:.0f} MiB", xy=(peak_t, peak), xytext=(6, 6),
                    textcoords="offset points", color="crimson", fontsize=9)

    if phases:
        span = ts[-1] if ts else 0.0
        for lab, f in phases:
            x = span * f
            ax.axvline(x, ls=":", lw=1.2, color="dimgray")
            ax.annotate(lab, xy=(x, ax.get_ylim()[1] * 0.965), xytext=(3, 0),
                        textcoords="offset points", fontsize=8.5, color="dimgray")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("allocated (MiB)")
    ax.set_title(title)
    ax.grid(alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("snapshot")
    p.add_argument("--plot", default=None, help="输出 PNG 路径")
    p.add_argument("--top", type=int, default=0, help="打印最大的 N 次分配及其栈回溯")
    p.add_argument("--phase-frac", nargs=2, type=float, default=None,
                   metavar=("FORWARD_END", "BACKWARD_END"),
                   help="阶段分界（占整段时长的比例，用于在图上标注）")
    p.add_argument("--title", default=None)
    p.add_argument("--total-ms", type=float, default=None,
                   help="实测的单步墙钟时间；给了就把 x 轴线性归一化到它。"
                        "（快照里的 time_us 不是墙钟——排查见 writeup 附录；"
                        "但它的**比例**与实测阶段占比一致，故线性归一后可用）")
    args = p.parse_args()

    snap = load(args.snapshot)
    ts, ys, peak, peak_t = build_timeline(snap)
    if not ts:
        print("快照里没有事件")
        return 1

    raw_span = ts[-1]
    if args.total_ms:                      # 线性归一到实测墙钟（仅供直观，刻度不严谨）
        k = args.total_ms / raw_span
        ts = [t * k for t in ts]
        peak_t *= k
        xlabel = "time (ms，按实测步时线性归一；注意快照时钟是 CPU 侧进度)"
    else:                                  # 默认：相对进度 %（最不容易误导）
        ts = [t / raw_span * 100 for t in ts]
        peak_t = peak_t / raw_span * 100
        xlabel = "相对进度（%，快照时钟）"

    # 阶段边界：优先用快照自带的 external_annotations（精确），否则用命令行给的比例
    phases = phase_markers(snap)
    if phases:
        print("阶段边界（快照时钟）: " + ", ".join(f"{n} {f * 100:.1f}%" for n, f in phases))
    elif args.phase_frac:
        phases = list(zip(("forward", "backward"), args.phase_frac))
        span = ts[-1]
        print(f"阶段估算  : forward 结束 ≈ {span * args.phase_frac[0]:.1f} ms, "
              f"backward 结束 ≈ {span * args.phase_frac[1]:.1f} ms")

    if args.top:
        print(f"\n--- 最大的 {args.top} 次分配 ---")
        for i, (size, frames) in enumerate(top_allocations(snap, args.top), 1):
            print(f"\n[{i}] {size / MiB:8.2f} MiB ({size:,} bytes)")
            for line in py_frames(frames):
                print(f"      {line}")

    if args.plot:
        out = Path(args.plot)
        out.parent.mkdir(parents=True, exist_ok=True)
        plot(ts, ys, out, args.title or out.stem, phases, peak, peak_t, xlabel)
        print(f"\n已写出图: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
