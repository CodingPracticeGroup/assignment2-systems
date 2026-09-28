# CS336 Assignment 2 (systems) — Writeup

> **环境**：2× RTX 4060 Ti 16GB（CC 8.9 / Ada，**无 NVLink**）· 驱动 595.91.07 · CUDA 13.0 · torch 2.11.0+cu130 · nsys 2025.3.2<br>
> **代码**：`cs336_systems/benchmark.py`、`cs336_systems/profile_nsys.py`、`cs336_systems/attention_nvtx.py`<br>
> **数据**：`profiles/REPORT.md`（nsys 自动汇总）、`profiles/baseline.json`（timeit 数字）
>
> ⚠️ **本机约束（影响部分小问的作答范围）**：只有 16 GB 显存，讲义 Table 1 里的
> `large` / `xl` / `10B` 装不下（`large` 连 forward-only 都 OOM，原因见 `benchmarking_script` (b)），
> 因此凡涉及"所有模型规模"的小问，只能给出 `small` / `medium` 的结果，并注明 OOM。

---

# Problem (benchmarking_script): Benchmarking Script (4 points)

## 题目（原文）

> **(a)** Write a script to perform basic end-to-end benchmarking of the forward pass, backward pass, and optimizer step in your model. Specifically, your script should support the following:
> - Given hyperparameters (e.g., number of layers), initialize a model.
> - Generate a random batch of data.
> - Run $w$ warm-up steps (before you start measuring time), then time the execution of $n$ steps (either only forward, forward and backward, or forward and backward with optimizer step, depending on an argument). For timing, you can use the Python `timeit` module (e.g., either using the `timeit` function, or using `timeit.default_timer()`, which gives you the system's highest resolution clock, thus a better default for benchmarking than `time.time()`).
> - Call `torch.cuda.synchronize()` after each step.
>
> _Deliverable:_ A script that will initialize a basics Transformer model with the given hyperparameters, create a random batch of data, and time forward-only, forward-and-backward, and full training steps that include the optimizer step.
>
> **(b)** Time the forward, backward, and optimizer step for the model sizes described in Section 2.1.2. Use 5 warmup steps and compute the average and standard deviation of timings over 10 measurement steps. How long does a forward pass take? How about a backward pass? Do you see high variability across measurements, or is the standard deviation small?
>
> _Deliverable:_ A 1-2 sentence response with your timings.
>
> **(c)** One caveat of benchmarking is not performing the warm-up steps. Repeat your analysis without the warm-up steps. How does this affect your results? Why do you think this happens? Also try to run the script with 1 or 2 warm-up steps. Why might the result still be different?
>
> _Deliverable:_ A 2-3 sentence response.

---

## (a) 脚本

**`cs336_systems/benchmark.py`**。要点：

| 要求 | 实现 |
|---|---|
| 按超参初始化模型 | `--model-size {small,medium,large,xl,10B}`，取讲义 §2.1.2 的 `(d_model, d_ff, num_layers, num_heads)`；另可 `--vocab-size` / `--context-length` / `--batch-size` 覆盖 |
| 随机 batch | `torch.randint` 生成 `(batch, ctx)` 的 token 与 target（只测速度/显存，不需要真实数据） |
| $w$ 步 warmup + $n$ 步计时 | `--warmup`（默认 5）/ `--steps`（默认 10） |
| 三种模式 | `--mode {forward, forward-backward, full}`，`full` = forward + loss + backward + `optimizer.step()` |
| 计时 | `timeit.default_timer()`，每个计时步**前后各一次** `torch.cuda.synchronize()` |
| 输出 | 一行一条 `RESULT model=… mode=… mean_ms=… std_ms=… peak_gb=…`，便于批量拼表 |

设计上两个刻意的选择：

1. **只有 `full` 模式才构造 optimizer** —— Adam 的 m/v 要 2 倍参数显存，不建它就能让 forward-only 模式测到更大的模型。
2. **所有变化点走命令行**（讲义 §2.1.3 明确建议）—— 因为后面会反复改精度、换层，第 2 个 problem `benchmarking_mixed_precision` 立刻就要用（已预留 `--dtype {fp32,bf16}`）。

---

## (b) 各模型规模的计时

**配置**：batch = 4，vocab = 10000，**fp32**，**5 warmup + 10 计时步**（时间 = 均值 ± 标准差，单位 ms）。
「backward」= `fwd+bwd` − `forward`；「optimizer」= `full` − `fwd+bwd`。

| 规模 | forward | fwd + bwd | full（含 optimizer） | → backward | → optimizer | 峰值显存 |
|---|---:|---:|---:|---:|---:|---:|
| **small**（128.6 M，ctx 512） | 86.66 ± 0.04 | 277.09 ± 0.11 | 316.11 ± 0.20 | **190.4** | 39.0 | 5.02 GB |
| **medium**（423.2 M，ctx 128） | 51.01 ± 0.06 | 165.21 ± 0.30 | 294.75 ± 0.31 | **114.2** | **129.5** | 7.93 GB |
| **medium**（ctx 256） | 104.01 ± 0.05 | 334.59 ± 0.33 | 462.72 ± 0.35 | **230.6** | **128.1** | 8.49 GB |
| **medium**（ctx 512） | 264.82 ± 0.12 | 837.86 ± 0.60 | 965.37 ± 0.40 | **573.0** | **127.5** | 13.72 GB |
| large / xl / 10B | ❌ **OOM** | ❌ | ❌ | — | — | 需要 ≥14.5 / 50.8 / 191 GB |

### 回答

**forward 多久、backward 多久**：以 `small`（ctx 512）为例，**forward 86.7 ms、backward 190.4 ms（≈ 2.2× forward）**，一个完整训练步 316.1 ms。`medium`（ctx 512）是 forward 264.8 ms、backward 573.0 ms，同样是 **2.2×**。

**变异性**：**标准差极小**。10 次测量的标准差只有 **0.04 ~ 0.60 ms（相对 0.02% ~ 0.2%）**，所以答案是「**没有明显变异**」——因为固定了频率、预热充分、且每一步前后都 `synchronize()`。

**两个从表里读出来的规律**：

1. **backward ≈ 2.2 × forward**（三个配置分别是 2.20 / 2.24 / 2.16 倍），与"反向计算量约为前向两倍"的预期一致。
2. ⭐ **optimizer 的时间与 context 完全无关**（medium 在 ctx 128/256/512 下分别是 129.5 / 128.1 / 127.5 ms，几乎不动）——**因为 Adam 的代价只取决于参数量，而 forward/backward 随 context 增长**。所以 context 越长，optimizer 在整步里的占比越小。

### ⚠️ 关于 large / xl / 10B 的 OOM（记录原因）

`large` **连 forward-only 都 OOM**，栈顶是：

```
cs336_basics/model.py:427  attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key")
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 80.00 MiB.
```

**死因是 attention 的 L×L 分数矩阵**：`small` 在 ctx 512 时峰值 5.0 GB、ctx 1024 时 10.9 GB、ctx 2048 就爆——**显存随 context 平方增长**。这**正是 §4 要用手写 FlashAttention-2 修掉的问题**（分块 + 在线 softmax，永不物化 L×L 矩阵），写完 FA2 后 `large` 就能装下。

---

## (c) 不做 warmup 会怎样

**实验**（`small`，ctx 512，固定 10 个计时步，只改 warmup 步数）：

| warmup | forward 均值 | forward **标准差** | full 均值 | full **标准差** |
|---:|---:|---:|---:|---:|
| **0** | 97.25 ms（+11.8%） | **32.53 ms**（↑**400×**） | 330.34 ms（+4.5%） | **48.52 ms**（↑**220×**） |
| 1 | — | — | 315.32 ms | 0.364 ms |
| 2 | 86.89 ms | 0.069 ms | 315.62 ms | 0.195 ms |
| **5** | 86.97 ms | 0.080 ms | 315.99 ms | 0.220 ms |

### 回答

**不做 warmup，被破坏的主要不是均值、而是可信度**：均值只高了 4.5%~11.8%，但**标准差从 0.2 ms 暴涨到 48.5 ms（220 倍）**——第一次迭代包含 cuBLAS/cuDNN 的 kernel 选择（heuristic）、CUDA module 加载、显存分配器首次向驱动申请，这些一次性开销让那一次远超稳态（实测第一次 forward **217.9 ms vs 稳态 8.6 ms，差 25 倍**），于是均值被拉高、方差被彻底污染。

**为什么 1~2 步之后结果仍会略有不同**：1 步之后最大的初始化开销已摊掉（均值 315.3 已很接近稳态 316.0），但**标准差仍是 5 步时的 1.7 倍**（0.364 vs 0.220）——10 次测量里只要有一次撞上分配器扩容或时钟波动，`std` 就会被显著拉高；到 2 步时才降到 0.195。所以 warmup 的作用不只是"让均值准确"，更是"**让方差收敛**"；本机实测 **2 步基本够，5 步更稳**。

---
---

# Problem (nsys_profile): Nsight Systems Profiling (5 points)

## 题目（原文）

> Profile your forward pass, backward pass, and optimizer step using `nsys` with two model sizes from Table 1 of your choice as well as three power-of-two context lengths larger than 128, where the largest available size should be the longest context length you can fit in memory. Pick the combinations you think would be the most interesting to look at. For each profile answer the following questions:
>
> **(a)** What is the total time spent on your forward pass? Does it match what we had measured before with the Python standard library?
>
> **(b)** What CUDA kernel takes the most cumulative GPU time during the forward pass? How many times is this kernel invoked during a single forward pass of your model? Is it the same kernel that takes the most runtime when you do both forward and backward passes? (Hint: look at the "CUDA GPU Kernel Summary" under "Stats System View", and filter using NVTX ranges to identify which parts of the model are responsible for which kernels.)
>
> **(c)** Although the vast majority of FLOPs take place in matrix multiplications, you will notice that several other kernels still take a non-trivial amount of the overall runtime. What other kernels besides matrix multiplies do you see accounting for non-trivial CUDA runtime in the forward pass?
>
> **(d)** Profile running one complete training step with your implementation of AdamW (i.e., the forward pass, computing the loss and running a backward pass, and finally an optimizer step, as you'd do during training). How does the fraction of time spent on matrix multiplication change, compared to doing inference (forward pass only)? How about other kernels?
>
> **(e)** Compare the runtime of the softmax operation versus the matrix multiplication operations within the self-attention layer of your model during a forward pass. How does the difference in runtimes compare to the difference in FLOPs?

## 我的 profile 配置

按"最大 context 取显存能装下的最长"的要求，先用 `probe` 摸底：

| 规模 | 能装到的最大 context | 该 context 下的峰值显存 |
|---|---:|---:|
| **small**（128.6 M） | **1024** | 10.89 GB |
| **medium**（423.2 M） | **512** | 13.72 GB |

于是取：**small @ 256 / 512 / 1024**，**medium @ 128 / 256 / 512**（各 3 个 2 的幂），每个配置都跑 **forward-only** 与 **full** 两份 profile，共 12 份。
跳过 warmup 的方式：`nsys -c cudaProfilerApi --capture-range-end=stop` + 代码里的 `torch.cuda.profiler.start/stop`。

---

## (a) forward 的总时间，与 timeit 是否吻合

**吻合。** `nvtx_gpu_proj_sum` 里 `:measure` 的 GPU 投影时间 ÷ 计时步数，与 `timeit` 的单步均值对照：

| 配置 | nsys `:measure` ÷ 步 | timeit 单步均值 | 误差 |
|---|---:|---:|---:|
| small ctx256 forward | 31.9 ms | 31.9 ms | **0.0%** |
| small ctx512 forward | 86.8 ms | 87.0 ms | 0.2% |
| small ctx1024 forward | 240.4 ms | 240.6 ms | 0.1% |
| medium ctx128 forward | 49.7 ms | 51.0 ms | 2.5% |
| small ctx512 full | 316.3 ms | 316.6 ms | 0.1% |
| medium ctx512 full | 965.1 ms | 967.0 ms | 0.2% |

### 回答

nsys 测到的 forward GPU 时间与 Python `timeit` 的结果**吻合到 0.1%–2.5%**（多数 < 0.5%），说明 ①profiling 开销可忽略、②`torch.cuda.synchronize()` 用法正确、③在 GPU-bound 场景下 `timeit` 测到的确实就是 GPU 执行时间。

> ⚠️ 口径提醒：`nvtx_gpu_proj_sum` 的 `:measure` 是**所有计时步的总和**，必须**除以步数**才能与 timeit 的单步均值比较（直接比会误得"nsys 慢 3 倍"）。

---

## (b) forward 里累计 GPU 时间最多的 kernel

以 **small / ctx 512** 与 **medium / ctx 512** 为例（数据来自 `nsys stats --report cuda_gpu_kern_sum:nvtx-name`，用 `forward/` 前缀把 forward 段筛出来）：

| 规模 | forward 里第一名 | 占比 | **单次 forward 的调用次数** |
|---|---|---:|---:|
| small（12 层） | `ampere_sgemm_64x64_tn` | 24.7% | **25 次** |
| medium（24 层） | `ampere_sgemm_64x64_tn` | 26.7% | **49 次** |

### 回答

**forward 里累计 GPU 时间最多的是 cuBLAS 的 `ampere_sgemm_64x64_tn`**（占比约 25%），单次 forward 里被调用 **`2 × num_layers + 1` 次**（small 的 12 层 → **25 次**；medium 的 24 层 → **49 次**；`+1` 是最后的 LM head）。**但它在完整训练步里不再是第一名** —— 加上 backward 后榜首变成 **`ampere_sgemm_64x64_nn`**（small 12.3% / medium 13.9%），而 `_tn` 掉到第 4 位（6.7% / 7.3%），因为反向传播引入的大量梯度矩阵乘使用了**转置形式不同的 cuBLAS kernel**。

*(`nn` / `tn` / `nt` 是 cuBLAS 对两个操作数是否转置的命名；同一层在前向与反向会落到不同变体上。)*

---

## (c) 除矩阵乘外，还有哪些 kernel 占了不可忽略的时间

**small / ctx 512 / forward**，非 matmul kernel（按累计 GPU 时间排序）：

| 占比 | 次数 / 单次 forward | kernel | 来自哪里 |
|---:|---:|---|---|
| **8.1%** | 24 | `vectorized_elementwise_kernel<BinaryFunctor<float,float,…>>` | 逐元素乘/除（attention 缩放、SwiGLU 门控乘） |
| **5.6%** | 12 | `vectorized_elementwise_kernel<exp_kernel_cuda>` | **softmax 的 `exp`** |
| 5.5% | 12 | `vectorized_elementwise_kernel<BUnaryFunctor<…>>` | 逐元素一元运算（dropout / 取负） |
| 4.4% | 12 | `elementwise_kernel<…BinaryFunctor…>` | 逐元素二元（非向量化路径） |
| **4.3%** | 12 | `elementwise_kernel<…CUDAFunctor_add<float>>` | **残差相加**（`x + attn_out`、`x + mlp_out`） |
| **4.1%** | 12 | `elementwise_kernel<…where_kernel_impl…>` | **因果 mask 的 `torch.where`** |
| **3.5%** | 12 | `reduce_kernel<…MaxOps<float>…>` | **softmax 的 max（数值稳定）** |
| **3.5%** | 12 | `reduce_kernel<…func_wrapper_t<float,…>>` | **softmax / LayerNorm 的 sum** |

> 以上合计约 **39%**，即 **forward 里约四成 GPU 时间花在非 matmul 上**。

### 回答

除矩阵乘外，占大头的是一整类**逐元素（elementwise）与归约（reduce）kernel**，且它们**每层各出现一次**（次数都是 12 = 层数）：其中与 **softmax 直接相关**的 `exp` + `max` + `sum` + `where`(mask) 四项合计约 **16.7%**，另有**残差相加**（`CUDAFunctor_add`）4.3% 与各种逐元素乘/除约 18%。这些算子算术强度极低、**完全受显存带宽约束**（对照 Lecture 5 的 Roofline：它们都落在斜率为带宽的斜坡区），因此虽然 FLOPs 占比极小，时间占比却很高。

---

## (d) 完整训练步里 matmul 占比怎么变

**6 组配置，forward-only vs 完整训练步**（占比 = matmul 类 kernel 累计 GPU 时间 ÷ 全部 kernel 累计 GPU 时间）：

| 模型 | context | forward matmul% | full matmul% | 变化 |
|---|---:|---:|---:|---:|
| medium | 128 | **84.6%** | 39.5% | **−45.0 pt** |
| medium | 256 | 74.9% | 47.6% | −27.3 pt |
| medium | 512 | 58.4% | 45.2% | −13.2 pt |
| small | 256 | 72.5% | 45.7% | −26.8 pt |
| small | 512 | 53.0% | 41.3% | −11.7 pt |
| small | 1024 | 41.1% | 35.3% | −5.9 pt |

**full 里新出现的冠军 kernel**（small ctx512）：

| 排名 | kernel | 占比 | 次数/步 |
|---:|---|---:|---:|
| 1 | `ampere_sgemm_64x64_nn` | 12.3% | 85 |
| 2 | `vectorized_elementwise_kernel<BinaryFunctor…>`（梯度逐元素） | 12.0% | **196** |
| 3 | `cutlass::Kernel2<cutlass_80_simt_sgemm_128x128_8x4_nt_align1>` | 7.8% | 37 |

另有一批 `multi_tensor_apply_kernel`（**AdamW** 的 multi-tensor 实现），在 medium 上合计约 20%。

### 回答

**matmul 占比大幅下降：从推理的 41%–85% 掉到训练步的 35%–48%**（最多下降 45 个百分点）。原因是反向传播与 optimizer 新增了大量**非 matmul** kernel——梯度逐元素运算（`BinaryFunctor` 的调用次数从 forward 的 24 次/步飙到 **196 次/步**）、`cutlass` 的转置矩阵乘变体，以及 AdamW 的 `multi_tensor_apply`（可达 ~20%）。**换句话说，forward 越"像 matmul"，backward + optimizer 就越"不像"**——这也说明只看端到端总时间是看不出这个结构性变化的。

> ⭐ **还有一个更值得注意的规律**：看 forward 那一列，**context 越长，forward 本身就越"不像 matmul"**（medium 84.6% → 58.4%；small 72.5% → 41.1%）。因为 attention 的分数矩阵是 **L²**，随 context 平方增长，而 attention 里**全是非 matmul 的活**（softmax 的 exp/max/sum、mask、逐元素）。**这正是 §4 要用 FlashAttention-2 优化的方向**：长上下文场景下瓶颈越来越不在 matmul。

---

## (e) self-attention 里 softmax vs 矩阵乘：runtime 与 FLOPs 的对比

**方法**：按讲义 §2.1.4 的做法，用猴子补丁把 `cs336_basics.model.scaled_dot_product_attention` 换成**带 NVTX 分段**的等价实现（计算逻辑一字未改，只加区间），见 `cs336_systems/attention_nvtx.py`；用 `--annotate-attention` 启用。

**实测**（small / ctx 512 / forward，3 个计时步，区间实例数 36 = 12 层 × 3 步）：

| 区间 | GPU 投影时间（3 步） | 每次调用 |
|---|---:|---:|
| **`attn: softmax`** | **55.16 ms** | 1.532 ms |
| `attn: QK^T matmul` | 21.22 ms | 0.589 ms |
| `attn: PV matmul` | 12.22 ms | 0.339 ms |

**FLOPs**（每层，batch `b`=4、序列 `s`=512、`d_model`=768、heads `a`=12）：

| 部件 | 公式 | FLOPs |
|---|---|---:|
| QK^T | `2·b·s²·d_model` | 1.611 × 10⁹ |
| PV | `2·b·s²·d_model` | 1.611 × 10⁹ |
| **两个 matmul 合计** | `4·b·s²·d_model` | **3.221 × 10⁹** |
| **softmax** | `≈ 5·a·b·s²`（max / sub / exp / sum / div） | **6.29 × 10⁷** |

### 回答

**runtime 的差距与 FLOPs 的差距完全不成比例，方向甚至是反的**：两个矩阵乘合计拥有 **51 倍的 FLOPs**（3.22 × 10⁹ vs 6.29 × 10⁷），却只用掉 **33.4 ms** GPU 时间；而 softmax 一个算子就用了 **55.2 ms**——**softmax 比两个 matmul 加起来还慢 1.65 倍**。折算成单位 FLOP 的效率，**softmax 比矩阵乘差约 85 倍**（`51.2 ÷ 0.606`）。

原因与 Roofline 模型一致：**矩阵乘是 compute-bound**（算术强度高，能喂饱 tensor core），而 **softmax 是彻底的 memory-bound**——它对 L×L 矩阵做多次遍历（求 max、减、exp、求和、除），每个元素只摊到个位数 FLOP，时间全花在 HBM 往返上。这既是讲义 (c)(d) 中"非 matmul kernel 占比很高"的微观解释，也说明**优化 attention 不应只盯着 matmul**。

> 注：softmax 的 "5 ops/element" 是常用近似（max / sub / exp / sum / div）；即使按 4 或 6 计，结论（差约两个数量级）不变。

---
---

# 附：本机环境、已知坑与复现

## 为什么 `-c nvtx` 用不了（记录备查）

讲义建议用 `nsys -c nvtx -p <range>` 跳过 warmup，但**本机实测完全触发不了采集**（试了 `measure` / `:measure` / `*measure*` / `measure@*` 四种写法、`uv run` 与直接调 python，**全部产出 0 字节**）。改用另一条官方路径：**CUDA Profiler API**（`torch.cuda.profiler.start/stop` + `-c cudaProfilerApi`），可用。

## NVTX 归因的两个陷阱（决定了 (b)(d) 的作答方式）

1. **`:backward` 的投影≈0**：PyTorch 的 autograd 引擎在**独立 device 线程**发射 kernel，主线程推的 `nvtx.range("backward")` 盖不住它们（实测只归因到 1 个 kernel，而 **34 种 kernel / 936 ms 完全未被任何区间覆盖**）。加 `--pytorch=autograd-nvtx` 能覆盖，但区间会细到**每个算子一个**（`aten::_foreach_addcdiv_, op_id=18667`），无法聚合成"backward 总共多少"。
   → **改用「forward-only 与 full 两份 profile 的差值」**，即 (d) 表的用法。
2. **`:optimizer` 被高估**：`optimizer.step()` 在 CPU 上紧接着 `backward()` 返回就执行，而 backward 的 GPU kernel 还在飞，**反向的尾巴被算进了 optimizer 窗口**（实测 medium ctx128 的 `:optimizer` 显示 395 ms，比整个 forward 的 147 ms 还大）。
   → **只有 `:measure`（整步）与 `:forward`（每步第一段）可信**。这也是 (e) 能成立的原因：attention 的各段在 forward 内部、主线程上，归因正常。

## 其他环境限制

| 项 | 情况 | 影响 |
|---|---|---|
| `kernel.perf_event_paranoid = 4` | `perf_event_open: Fail` → 警告 "CPU IP/backtrace sampling not supported" | **不影响 (a)–(e)**（都要 GPU 侧数据）；但 `--cudabacktrace` / `--python-backtrace` 拿不到数据 |
| 无 NVLink（消费级 Ada） | 多卡通信走 PCIe | 影响后续 §5 起的 DDP/FSDP 数字 |
| `nsys stats` 的 sqlite 缓存 | 输入比缓存新时会拒绝运行、打印 usage，**但退出码仍是 0** | 永远加 `--force-export=true` |

## 复现

```bash
cd assignment2-systems
export PATH=/usr/local/cuda-13.0/bin:$PATH

# benchmarking_script (b)(c)
uv run python -m cs336_systems.benchmark --model-size small --context-length 512 --mode full --warmup 5 --steps 10
for w in 0 1 2 5; do
  uv run python -m cs336_systems.benchmark --model-size small --context-length 512 --mode full --warmup $w --steps 10
done

# nsys_profile：probe → profile → baseline → report
uv run python -m cs336_systems.profile_nsys probe    --model-sizes small medium
uv run python -m cs336_systems.profile_nsys profile  --model-sizes small  --context-lengths 256 512 1024
uv run python -m cs336_systems.profile_nsys profile  --model-sizes medium --context-lengths 128 256 512
uv run python -m cs336_systems.profile_nsys baseline --model-sizes small medium --context-lengths 128 256 512 1024
uv run python -m cs336_systems.profile_nsys report   --outdir profiles --steps 3

# nsys_profile (e)：带 attention 内部分段的 profile
nsys profile --force-overwrite=true -o profiles/attn_small_ctx512_forward \
  --trace=cuda,nvtx -c cudaProfilerApi --capture-range-end=stop -- \
  uv run python -m cs336_systems.benchmark --model-size small --context-length 512 \
  --mode forward --warmup 2 --steps 3 --nvtx --profiler-range --annotate-attention
nsys stats --force-export=true --report nvtx_gpu_proj_sum profiles/attn_small_ctx512_forward.nsys-rep
```
