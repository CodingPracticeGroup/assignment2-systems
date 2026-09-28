# memory_profiling 交付物索引

讲义 assignment 2 §2.2.6 的 (a)(b)(c)(e)(f) 用的显存快照与时间线。

**`.pickle` 不入库**（单份 1–2 MB，共 19 MB）——它们是 `torch.cuda.memory._dump_snapshot`
的原始产物，可用下面一行命令按需重新生成；**`.png` 入库**，是 (a)/(f) 的时间线交付物。

## 图

每张 `<name>.png` 由 `memory_report.py` 从同名 `.pickle` 重建成
**Active Memory Timeline**（等价于 `pytorch.org/memory_viz`，但可版本化、可自动化）：
y 轴为已分配显存（含录制前的参数基线），x 轴为**相对进度 %**
（快照的 `time_us` 不是墙钟，见 writeup 附录），竖线是 `external_annotations`
里 `PHASE_forward` / `PHASE_backward` / `PHASE_optimizer` 的精确边界。

## 峰值对照表（GiB，含基线；已与 `torch.cuda.max_memory_allocated()` 逐位对齐）

| 快照 | forward | full step |
|---|---:|---:|
| `small_ctx128_fp32` | 1.11 | 2.43 |
| `small_ctx512_fp32` | 3.82 | 5.02 |
| `small_ctx1024_fp32` | 9.41 | 10.88 |
| `small_ctx512_bf16` | 2.92 | 4.07 |
| `medium_ctx128_fp32` | 3.21 | 7.92 |
| `medium_ctx256_fp32` | 5.20 | 8.48 |
| `medium_ctx512_bf16` | 8.15 | 11.52 |

`xl`（3.407 B 参数 = 12.69 GiB）与 ctx 2048 在本机 16 GB 卡上不可行，定量原因见 writeup 的
「⚠️ 前置」小节。

## 重新生成

```bash
# 一份快照
uv run python -m cs336_systems.benchmark --model-size small --context-length 512 \
  --mode full --dtype fp32 --warmup 5 --steps 1 \
  --memory-snapshot profiles/memory/small_ctx512_fp32_full.pickle

# 重建时间线
uv run python -m cs336_systems.memory_report profiles/memory/small_ctx512_fp32_full.pickle \
  --plot profiles/memory/small_ctx512_fp32_full.png --title "small ctx512 fp32 full"

# 最大分配的栈回溯（(e)/(f) 的操作归因）
uv run python -m cs336_systems.memory_report profiles/memory/small_ctx512_fp32_forward.pickle --top 6
```
