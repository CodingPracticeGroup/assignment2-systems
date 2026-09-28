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

# Problem (mixed_precision_accumulation): Mixed-Precision Accumulation (1 point)

## 题目（原文）

> Run the following code and comment on the accuracy of the results.
>
> ```python
> s = torch.tensor(0,dtype=torch.float32)
> for i in range(1000):
>     s += torch.tensor(0.01,dtype=torch.float32)
> print(s)
> s = torch.tensor(0,dtype=torch.float16)
> for i in range(1000):
>     s += torch.tensor(0.01,dtype=torch.float16)
> print(s)
> s = torch.tensor(0,dtype=torch.float32)
> for i in range(1000):
>     s += torch.tensor(0.01,dtype=torch.float16)
> print(s)
> s = torch.tensor(0,dtype=torch.float32)
> for i in range(1000):
>     x = torch.tensor(0.01,dtype=torch.float16)
>     s += x.type(torch.float32)
> print(s)
> ```
>
> _Deliverable:_ A 2-3 sentence response.

## 实测结果

脚本：`cs336_systems/mixed_precision_accumulation.py`（循环 1000 次、每次加 0.01，数学精确值 = **10**）。

| 方案 | 结果 | 绝对误差 | 相对误差 |
|---|---:|---:|---:|
| **①** `s=fp32, 加数=fp32` | 10.000134 | +0.000134 | +1.3 × 10⁻⁵ |
| **②** `s=fp16, 加数=fp16` | **9.953125** | **−0.046875** | **−4.7 × 10⁻³** |
| **③** `s=fp32, 加数=fp16` | 10.002136 | +0.002136 | +2.1 × 10⁻⁴ |
| **④** `s=fp32, 加数 fp16→fp32` | **10.002136** | +0.002136 | +2.1 × 10⁻⁴ |

关键的中间量：

| 量 | 值 |
|---|---:|
| `0.01` 在 fp32 里的实际值 | 0.009999999776（误差 −2.2 × 10⁻¹⁰） |
| `0.01` 在 fp16 里的实际值 | **0.010002136230**（误差 **+2.1 × 10⁻⁶**） |
| **1000 × fp16 的加数误差** | **+0.002136** ← 与 ③④ 的误差**完全相等** |
| fp16 在 10 附近的 ULP | 0.0078125（略小于 0.01，所以还能加上去，但每步都被舍入） |

### 回答

**四个结果分成三档**：全 fp32（①）误差仅 1.3 × 10⁻⁵，是可信基准；**把累加器也放到 fp16（②）结果严重偏低（9.953，误差 −0.047）**，因为每一步 `s += 0.01` 都在 fp16 里舍入（在 s ≈ 10 处 ULP 已达 0.0078，每步的舍入误差不断累积）；而**累加器保持 fp32、只把加数降到 fp16（③④）误差小一个数量级（+0.0021）**。

**③ 与 ④ 的结果逐位相同** —— 说明 PyTorch 的 **type promotion 已经自动把 fp16 加数提升到 fp32 再相加**，显式 `.type(torch.float32)` 是数值上的 no-op。而且 **③④ 的误差恰好等于 `1000 × (fp16(0.01) − 0.01) = +0.002136`**，即误差 **100% 来自"0.01 本身存不进 fp16"这个表示误差**，累加过程没有额外放大它。

**结论（也是混合精度训练的核心规则）**：必须把「**累加器精度**」与「**加数精度**」分开——加数（权重/激活）可以降到 fp16/bf16，但**累加必须留在 fp32**，这正是 ③ 的模式、也正是 tensor core 用 fp16/bf16 做乘法却用 fp32 累加的原因；②与③的误差差 **22 倍**，说明**累加器精度是主导因素**。

---

# Problem (benchmarking_mixed_precision): Benchmarking Mixed Precision (2 points)

## 题目（原文）

> **(a)** Consider the following model:
>
> ```python
> class ToyModel(nn.Module):
>     def __init__(self, in_features: int, out_features: int):
>         super().__init__()
>         self.fc1 = nn.Linear(in_features, 10, bias=False)
>         self.ln = nn.LayerNorm(10)
>         self.fc2 = nn.Linear(10, out_features, bias=False)
>         self.relu = nn.ReLU()
>     def forward(self, x):
>         x = self.relu(self.fc1(x))
>         x = self.ln(x)
>         x = self.fc2(x)
>         return x
> ```
>
> Suppose we are training the model on a GPU and that the model parameters are originally in FP32. We'd like to use autocasting mixed precision with FP16. What are the data types of:
> - the model parameters within the autocast context?
> - the output of the first feed-forward layer (`ToyModel.fc1`)?
> - the output of layer norm (`ToyModel.ln`)?
> - the model's predicted logits?
> - the loss?
> - the model's gradients?
>
> _Deliverable:_ The data types for each of the components listed above.
>
> **(b)** You should have seen that FP16 mixed precision autocasting treats the layer normalization layer differently than the feed-forward layers. What parts of layer normalization are sensitive to mixed precision? If we use BF16 instead of FP16, do we still need to treat layer normalization differently? Why or why not?
>
> _Deliverable:_ A 2-3 sentence response.
>
> **(c)** Modify your benchmarking script to optionally run the model using mixed precision with BF16. Time the forward and backward passes with and without mixed-precision for each language model size described in Section 2.1.2. Compare the results of using full precision versus mixed precision, and comment on any trends as model size changes. You may find the `nullcontext` no-op context manager to be useful.
>
> _Deliverable:_ A 2-3 sentence response with your timings.

---

## (a) 各部件的 dtype（实测，非推断）

脚本：`cs336_systems/mixed_precision_dtypes.py`（用 `register_forward_hook` 抓算子输出、backward 后查 `.grad`）。

| 部件 | autocast 关闭 | **autocast(fp16)** | autocast(bf16) |
|---|---|---|---|
| 模型参数（在 autocast 上下文**内部**读） | float32 | **float32** | float32 |
| `ToyModel.fc1` 输出 | float32 | **float16** | bfloat16 |
| **`ToyModel.ln` 输出（LayerNorm）** | float32 | **float32** ⭐ | **float32** ⭐ |
| 模型预测 logits（`fc2` 输出） | float32 | **float16** | bfloat16 |
| **loss** | float32 | **float32** | float32 |
| 模型梯度 | float32 | **float32** | float32 |
| backward 之后的参数 | float32 | float32 | float32 |

### 回答

**参数、LayerNorm 输出、loss、梯度全部保持 float32；只有 `fc1` / `fc2` / logits 变成 float16。** 具体地：模型参数**在 autocast 内仍是 fp32**（autocast 不改参数本身，只在算子内部临时 cast 权重与输入）；`fc1` 输出 fp16（矩阵乘走 tensor core）；**`ln` 输出 fp32**（LayerNorm 在 autocast 的 fp32 白名单里）；logits 是 fp16（`fc2` 又是矩阵乘）；**loss 是 fp32**（`cross_entropy` 内部的 `log_softmax` 与 `nll_loss` 都在 fp32 名单里）；**梯度是 fp32**（反向把 fp16 中间结果的梯度累加回 fp32 参数）。

---

## (b) LayerNorm 哪些部分对混合精度敏感

`autocast(fp16)` 下实测的数值误差：

| 量 | 结果 |
|---|---|
| fp32 重算的 mean / var | +2.91854453 / +4.82945633 |
| **fp16 重算的 mean / var** | +2.91796875 / **+4.83203125** |
| → mean 绝对误差 / var 绝对误差 | 5.8 × 10⁻⁴ / **2.6 × 10⁻³** |
| → 归一化输出的最大差 | 2.0 × 10⁻³ |

| dtype | max | 最小正规数 | eps（尾数精度） |
|---|---:|---:|---:|
| **fp16** | **6.55 × 10⁴** | 6.10 × 10⁻⁵ | 9.8 × 10⁻⁴ |
| **bf16** | **3.39 × 10³⁸** | 1.18 × 10⁻³⁸ | 7.8 × 10⁻³ |
| fp32 | 3.40 × 10³⁸ | 1.18 × 10⁻³⁸ | 1.2 × 10⁻⁷ |

### 回答

LayerNorm 里有三处对 fp16 敏感：**① 归约量 mean/var** —— `var = E[x²] − E[x]²` 是"两个大数相减"，fp16 只有 10 位尾数，有效位损失被放大（实测 var 误差 2.6 × 10⁻³，比 mean 的误差大 4 倍，并经 `1/sqrt(var+eps)` 传给输出）；**② `eps`** —— 默认 `eps = 1e-5` **在 fp16 里根本无法表示**（fp16 最小正规数 6.1 × 10⁻⁵），会被舍成 0 或 6.1 × 10⁻⁵，直接改变数值稳定性；**③ 动态范围** —— fp16 的 `max` 只有 6.55 × 10⁴，归一化前的激活（尤其大模型、长序列）很容易越界成 `inf`。

**换成 bf16 则不需要特别对待**：bf16 的**指数位与 fp32 相同（8 位）**，动态范围 3.39 × 10³⁸ 与 fp32 完全一致，不会溢出（代价仅是尾数更少、`eps` 更大 7.8 × 10⁻³，但 LayerNorm 的主要风险是溢出而非精度）。**实测补充**：PyTorch 的 autocast 对 bf16 **也照样把 LayerNorm 放在 fp32 名单里**（输出仍是 float32）——这是保守的工程策略，而不是 bf16 的必要性。

---

## (c) full precision vs BF16 的实测对比

**配置**：batch = 4，context = 512，vocab = 10000，**5 warmup + 10 计时步**（均值 ± 标准差，单位 ms）。bf16 通过 `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`；不用 autocast 时用 `contextlib.nullcontext()`。

| 规模 | 模式 | fp32 | **bf16** | **加速比** | 峰值显存 fp32 → bf16 |
|---|---|---:|---:|---:|---|
| **small**（128.6 M） | forward | 86.55 ± 0.04 | 52.75 ± 0.05 | **1.64×** | 3.82 → 2.97 GB（−22%） |
| **small** | forward+backward | 276.91 ± 0.12 | 170.58 ± 0.06 | **1.62×** | 4.55 → 3.66 GB（−20%） |
| **medium**（423.2 M） | forward | 264.44 ± 0.07 | 153.67 ± 0.04 | **1.72×** | 10.33 → 8.16 GB（−21%） |
| **medium** | forward+backward | 838.39 ± 0.49 | 498.52 ± 0.04 | **1.68×** | 12.14 → 9.93 GB（−18%） |

`large` / `xl` / `10B` 在本机 16 GB 卡上 OOM（原因见前文 `benchmarking_script` (b)），未测。

### 回答

**BF16 混合精度带来约 1.6–1.7× 的加速，且加速比随模型变大而略微上升**（forward：small 1.64× → medium 1.72×），因为越大的模型矩阵乘占比越高，而矩阵乘正是 bf16 tensor core 受益最大的部分；**显存同步下降约 20%**（small 3.82 → 2.97 GB，medium 10.33 → 8.16 GB），来自激活值减半（参数与优化器状态仍是 fp32）。

**为什么只有 1.6–1.7×、而不是按 tensor core 峰值算出的好几倍？Amdahl 定律给出定量解释**：前一个 problem 实测出 forward 里**只有约 53%（small）/ 58%（medium）的时间是矩阵乘**，其余是 elementwise / softmax / LayerNorm 等不吃 tensor core 的算子，且 LayerNorm 与 loss 仍固定在 fp32。若假设矩阵乘在 bf16 下快 4×、非矩阵乘不变，则预测加速 = `1 / (0.53/4 + 0.47) = 1.66×`（medium 为 1.78×）——**与实测的 1.64× / 1.72× 吻合**。这也说明：**要进一步吃到低精度红利，必须先把那些非矩阵乘算子也优化掉**（正是 §4 FlashAttention 的动机）。

---
---

# Problem (memory_profiling): Memory Profiling (4 points)

## 题目（原文）

> Profile your complete training step of forward pass, backward pass, and optimizer step of the `xl` model from Table 1 with context lengths of 128 and 2048.
>
> **(a)** Add an option to your profiling script to run your model through the memory profiler. It may be helpful to reuse some of your previous infrastructure (e.g., to activate mixed-precision, load specific model sizes, etc). Then, run your script to get a memory profile of the `xl` model when either doing inference only (just forward pass) or a full training step. What do your memory timelines look like? Can you tell which stage is running based on the peaks you see?
>
> *Deliverable:* Two images of the "Active memory timeline" of an `xl` model, from the `memory_viz` tool: one for the forward pass, and one for running a full training step (forward and backward passes, then optimizer step), and a 2-3 sentence response.
>
> **(b)** What is the peak memory usage of each context length when doing a forward pass? What about when doing a full training step?
>
> *Deliverable:* A table with two numbers per context length.
>
> **(c)** Find the peak memory usage of the `xl` model when using mixed-precision, for both a forward pass and a full training step. Does mixed-precision significantly affect memory usage?
>
> *Deliverable:* A 2-3 sentence response.
>
> **(d)** Consider the `xl` model. Given our reference hyperparameters, what is the size of a tensor of activations in the Transformer residual stream, in single-precision? Give this size in MiB (i.e., divide the number of bytes by $1024^2$).
>
> *Deliverable:* A 1-2 sentence response with your derivation.
>
> **(e)** Now look closely at the "Active Memory Timeline" from `pytorch.org/memory_viz` of a memory snapshot of the `xl` model doing a forward pass. When you reduce the "Detail" level, the tool hides the smallest allocations to the corresponding level (e.g., putting "Detail" at 10% only shows the 10% largest allocations). What is the size of the largest allocations shown? Looking through the stack trace, can you tell where those allocations come from?
>
> *Deliverable:* A 1-2 sentence response.
>
> **(f)** Nsight Systems also has flags for memory profiling. You can combine these with the Nsight flags from before to understand what allocations are happening at different steps in your model's lifespan. Use the PyTorch-provided NVTX labels to determine how much memory is saved for backward (these tensors are often called residuals) by a single `TransformerBlock` in your model. Note the 5 largest contributing operations, and what percentage of the overall memory they contribute. During the backward pass, all these tensors will be freed, but new gradient tensors are emitted at the same time. Based on your profiles showing how much memory was allocated during the forward pass, and how much memory usage changes for every `TransformerBlock` in the backward pass, calculate how much memory the produced gradient tensors for a `TransformerBlock` take. Does the result match what you expect?
>
> *Deliverable:* Screenshots from Nsight Systems and a 1-2 paragraph response.

## ⚠️ 前置：`xl` 与 context 2048 在本机不可行（以及替代方案）

讲义要求 profile **`xl`**（$d_{model}=2560$、32 层、$d_{ff}=10240$）的 **ctx 128 与 2048**。在本机单张 16 GB 卡上这两个目标**都不可能**，原因可以精确算出来：

| 项 | 计算 | 结果 |
|---|---|---|
| `xl` 参数量 | 实测（`sum(p.numel())`） | **3.407 B** |
| 参数本身 fp32 | 3.407e9 × 4 B | **12.69 GiB** |
| 优化器（AdamW）状态 | 2 × 3.407e9 × 4 B | **25.4 GiB** |
| ctx 128 每层残差（解析粗算） | 3·B·H·S² + 4·B·S·d_ff + 6·B·S·d_model | 134 MiB → ×32 层 = **4.19 GiB** |
| ctx 2048 每层残差（解析粗算） | 同上，S² 项主导 | 7.90 GiB → ×32 层 = **247 GiB** |

- `xl` **只跑 forward** 就已经 $12.69 + 4.19 + 0.3 \approx 17.2$ GiB $> 16$ GiB → **OOM**（实测确认）。
- `xl` 跑 full training step 还要再加参数梯度 12.69 GiB + AdamW 状态 25.4 GiB ≈ **51 GiB**，是显存容量的 3 倍以上。
- ctx 2048 的彻底崩掉来自朴素 attention 的 $O(S^2)$：单份 `B·H·S²` 张量就是 **2.0 GiB**，softmax/`where`/`scores` 各留一份，单层就 6 GiB。**这正是讲义 §4 FlashAttention 要解决的问题**；顺带说明：在这个实现上，ctx 2048 不是"调小 batch"能救的。

**替代方案**（下面所有结论都标注了「实测」还是「解析外推」）：

1. **实测**：在能放下的最大配置上做完整测量 —— `small`（0.13 B，ctx 128 / 512 / 1024）与 `medium`（0.42 B，ctx 128 / 256 / 512-bf16）。ctx 取到 `small` 1024、`medium` 256，是因为再往上（`small` ctx 2048）朴素 attention 的 $B H S^2$ 项自己就超过 16 GB。
2. **解析外推**：把 `xl` 的公式量（(d)(e)）直接算出来，并给出 `xl` 会 OOM 的定量证据。

## (a) forward / full training step 的内存时间线

快照由新增的 `--memory-snapshot PATH` 选项产出（`benchmark.py` 内部调 `torch.cuda.memory._record_memory_history(max_entries=1_000_000)` → 跑计时循环 → `_dump_snapshot`）。

讲义要求的 `pytorch.org/memory_viz` 是个**网页工具**：图不能版本化、也不能自动化。所以 `cs336_systems/memory_report.py` 直接解析快照 pickle 里现成的 `device_traces`（每条含 `time_us` / `size` / `frames`），**重建出同一条 Active Memory Timeline** 并存成 PNG，同时把最大的若干次分配及其 **Python 栈回溯**打出来。产出图（`small` ctx128，fp32，batch 4）：

- forward：`profiles/memory/small_ctx128_fp32_forward.png`
- full：`profiles/memory/small_ctx128_fp32_full.png`

全部 14 张时间线在 `profiles/memory/*.png`（`small`/`medium` × fp32/bf16 × forward/full）。阶段分界不是猜的：`benchmark.py` 在每段外面套了 `record_function("PHASE_forward")` / `PHASE_backward` / `PHASE_optimizer`，`memory_report.phase_markers()` 读快照的 `external_annotations` 把它标在图上。

**先说两个踩到的坑**（决定了图怎么画）：

1. **录制前已分配的显存必须补回来**。`_record_memory_history` 只记录开启之后的事件，模型参数（`small` 490.7 MiB、`medium` 1622.5 MiB、full 模式下还包括 AdamW 状态）的 alloc **不在 `device_traces` 里**，直接用事件流回放会低估整整一个参数量。`memory_report.baseline_mib()` 反推：`dump 时活跃量 = 基线 + Σalloc − Σfree_completed`。补上基线后，**14 份快照的重放峰值全部与 `torch.cuda.max_memory_allocated()` 逐位吻合**（例：`small` ctx512 full 重放 5137.5 MiB vs `RESULT peak_gb=5.02`）。
2. **快照的 `time_us` 不是墙钟**。整条 trace 跨度只有 154 个时间单位，而这一步墙钟是 316 ms；`_dump_snapshot` 自己还花了约 1.1 s。而且它的 x 轴跟踪的是 **CPU 侧下发进度**（forward 占快照时钟的 5.5%，占墙钟却是 27.4%——因为 GPU 还在跑 backward 时 CPU 已经冲进了 optimizer 窗口）。所以 x 轴默认用**相对进度（%）**，阶段用 `external_annotations` 标注，不用墙钟比例。

### 回答

内存时间线呈**清晰的阶梯状**，**可以只凭形状判断当前处于哪个阶段**：forward 段是一条**单调上爬、斜率恒定**的斜坡（每层残差大小相同，所以是理想的等步长阶梯），到最高点平台；backward 段**逐层对称地把残差释放掉**，同时叠加上新的梯度分配，但**净效果是快速下降**，斜率接近 forward 的镜像；optimizer（AdamW）段是一条**几乎水平的直线**——`step()` 本身只分配极少量临时量（实测 `medium` ctx256 的 `optimizer.step()` 单独耗时 131.98 ms，但显存增量 ≈ 0），因为 AdamW 的 `exp_avg` / `exp_avg_sq` 在**第一次 step 时**就分配好了、之后一直常驻。三段的拐点非常干脆，依据是**前一段的净变化率符号**：持续上升 = forward，持续下降 = backward，持平 = optimizer。

## (b) 各 context length 的峰值显存（forward / full training step）

**实测，fp32，batch 4，vocab 10000，5 warmup + 10 计时步**。峰值 = 快照重放（基线 + 事件流），已与 `max_memory_allocated()` 对齐；单位 **GiB**（列里同时给出 MiB 便于对照 (d)）。

| 模型 | ctx | forward（GiB） | full step（GiB） | forward（MiB） | full（MiB） |
|---|---:|---:|---:|---:|---:|
| `small`（128.6 M） | 128 | **1.11** | **2.43** | 1135.9 | 2489.2 |
| `small` | 512 | **3.82** | **5.02** | 3913.6 | 5137.5 |
| `small` | 1024 | **9.41** | **10.88** | 9638.5 | 11142.5 |
| `medium`（423.2 M） | 128 | **3.21** | **7.92** | 3282.7 | 8107.4 |
| `medium` | 256 | **5.20** | **8.48** | 5327.8 | 8681.7 |
| `xl`（3.407 B） | 128 | *17.2（解析）* | *51（解析）* | *≈17600* | *≈52000* |
| `xl` | 2048 | *≈260（解析）* | *>300（解析）* | — | — |

**`xl` 那两行是解析外推，不是实测**（原因见前置小节）：`xl` forward @ ctx128 = 参数 12.69 GiB + 残差 4.19 GiB + logits 0.3 GiB ≈ **17.2 GiB > 16 GiB → OOM**；full step 再叠加参数梯度 12.69 GiB + AdamW 状态 25.4 GiB ≈ **51 GiB**；ctx 2048 因 $O(S^2)$ 直接到 **260 GiB 量级**。

### 回答

**峰值随 context 增长是超线性的**：`small` 从 ctx128 → 512 → 1024，forward 峰值 1.11 → 3.82 → 9.41 GiB，比值 3.4× / 2.5×，远高于 ctx 的 4× 线性增长（$S$ 变 4 倍，线性项只涨 4 倍，但 attention 的 $B H S^2$ 项涨 **16 倍**，于是它逐渐主导）。**full step / forward 的比值随模型变大而下降**：`small` ctx512 是 1.31×（5.02/3.82），`medium` ctx128 却是 2.47×（7.92/3.21）——原因是 `full` 多出来的是**参数梯度 + AdamW 状态**（对 `small` 约 1.4 GiB，对 `medium` 约 4.7 GiB），这部分**只随参数量走、与 context 无关**，而 `small` 的 forward 残差里 attention 项已经很大，把比值摊薄了。

## (c) 混合精度（BF16）对峰值显存的影响

**实测，batch 4**。bf16 用 `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`：

| 模型 | ctx | 模式 | fp32 峰值 | **bf16 峰值** | 变化 |
|---|---:|---|---:|---:|---:|
| `small` | 512 | forward | 3.82 GiB | **2.92 GiB** | **−24%** |
| `small` | 512 | full | 5.02 GiB | **4.07 GiB** | **−19%** |
| `medium` | 512 | forward | 10.33 GiB | **8.15 GiB** | **−21%** |
| `medium` | 512 | full | 13.72 GiB | **11.52 GiB** | **−16%** |

（`medium` ctx512 的 fp32 峰值取自 `profiles/baseline.json` 的计时跑，bf16 来自快照重放。）

### 回答

**混合精度对显存的影响是显著的，但远不到"减半"**：forward 段省 **21–24%**，full step 段只省 **16–19%**。原因很明确——autocast 只把**激活值**（矩阵乘的输入输出）降到 bf16，而**参数、参数梯度、AdamW 状态（`exp_avg` / `exp_avg_sq`）全部仍留在 fp32**。full step 比 forward 省得少，正是因为 full step 里那个"不缩水的大头"（梯度 + 优化器状态，对 `medium` 就是 3 × 423 M × 4 B ≈ 4.7 GiB）占比更高，稀释了激活值减半的收益。**推论：单卡上想进一步压显存，必须动优化器状态本身**（→ 后续 assignment 的 ZeRO / FSDP），这也和 (b) 里"full/forward 比值随参数量增大"的现象是同一枚硬币的两面。

## (d) residual stream 上一个 activation 张量的大小

**参考超参**：batch size 4、`xl` 的 $d_{model} = 2560$、fp32（4 B）。residual stream 的张量形状是 `(batch, seq_len, d_model)`，所以

$$
\text{bytes} = 4 \times S \times 2560 \times 4 = 40960 \times S
$$

- **ctx 128**：$4 \times 128 \times 2560 \times 4 = 5{,}242{,}880$ B $= 5{,}242{,}880 / 1024^2 =$ **5.0 MiB**
- **ctx 2048**：$4 \times 2048 \times 2560 \times 4 = 83{,}886{,}080$ B $/$ $1024^2 =$ **80.0 MiB**

### 回答

一个 residual stream 张量 = `batch × seq_len × d_model × 4 B`：**ctx 128 时 5.0 MiB，ctx 2048 时 80.0 MiB**。可见它**完全不是瓶颈**——`xl` 的真正显存杀手是 attention 的 $B H S^2$ 中间量（ctx2048 单份 2.0 GiB，是 residual stream 单份的 **25.6 倍**，而 $d_{model} = 32 \times 80 = H \cdot d_k$ 恰好解释了 $S/d_k = 2048/80 = 25.6$ 这个比例）。

## (e) 把 Detail 调低后，最大的那几次分配是什么

`memory_report.py --top N` 直接按大小排序并打印每次分配的**完整 Python 栈回溯**。

**实测（`small` ctx128 fp32 forward）**，最大的一次分配是 **19.53 MiB**，栈回溯指向

```
einsum(...)  ←  torch/functional.py:373
             ←  cs336_basics/model.py:39   Linear.forward:  einsum(x, W, "... d_in, d_out d_in -> ... d_out")
```

即 **LM head（output embedding 那个 `Linear`）的输出 logits**，形状 `(4, 128, 10000)`：$4 \times 128 \times 10000 \times 4 = 19.53$ MiB。紧随其后是一串 **6.00 MiB** 的分配，全部来自 `cs336_basics/model.py:532` 的 `silu`（`x * torch.sigmoid(x)`，形状 `(4, 128, 3072)`，$4 \times 128 \times 3072 \times 4 = 6.00$ MiB）。

### 回答

最大的分配是 **LM head 输出的 logits**（`small`/`medium`/`xl` 在 ctx128 下都是 **19.53 MiB**，因为它只取决于 `batch × ctx × vocab`，**与模型大小无关**），栈回溯落在 `cs336_basics/model.py:39` 的 `einsum` 上；往后是 MLP 的中间激活 `x * sigmoid(x)`（`small` 6.00 MiB）。**换成 `xl` 结论会翻转**：ctx128 下 $B S d_{ff} = 4 \times 128 \times 10240 \times 4 =$ **20.0 MiB**（`silu` / `w3(x)` / 两者乘积各一份），**略大于** 19.5 MiB 的 logits，此时最大分配变成 MLP 中间激活；而 ctx2048 下 attention weights 的 $B H S^2$ 项达到 **2048 MiB（2 GiB）**，比 logits（312.5 MiB）大一个数量级。

## (f) 单个 TransformerBlock 的残差、Top-5 贡献算子、以及反向的梯度显存

### 方法：用「层数斜率」精确测量单层开销

栈回溯里**所有 32 个 block 共用同一批 `model.py` 行号**，无法直接区分是第几层。所以改用更干净、也更严格的**控制变量法**：固定其它一切，只改 `num_layers`，测峰值，**斜率就是单个 TransformerBlock 的边际开销**。配置：`small`（$d_{model}=768$, $H=12$, $d_{ff}=3072$），ctx 512，batch 4，fp32，warmup 2 步后 `reset_peak_memory_stats()`：

| `num_layers` | forward-only 峰值 | full step 峰值 |
|---:|---:|---:|
| 6 | 2.068 GiB | 2.770 GiB |
| 9 | 2.984 GiB | 3.894 GiB |
| 12 | 3.900 GiB | 5.021 GiB |
| **斜率（每层增量）** | **312.6 MiB/block**（两次斜率 0.3052/0.3052 GiB，完全线性） | **384.2 MiB/block**（0.3746/0.3757） |

线性度极好（两次斜率小数第四位才分开），说明这个测法可信。实验落在 `cs336_systems/memory_per_block.py`：

```bash
uv run python -m cs336_systems.memory_per_block \
    --d-model 768 --num-heads 12 --d-ff 3072 --context-length 512 --batch-size 4 \
    --layer-counts 6 9 12
```

该模块同时把**解析对账**打出来（该层参数量 → 参数梯度尺寸 → 纯残差 → 临时激活梯度），下面的三个结果就是它的输出。

### 结果 1：单层为反向保存的残差 ≈ 277 MiB

**312.6 MiB 的边际增量里，36.01 MiB 是该层自己的参数**（实测每 block 参数 9.4387 M × 4 B = 36.01 MiB；其中 attention 的 Q/K/V/O = 9.00 MiB、FFN 的 w1/w2/w3 = 27.00 MiB、两个 RMSNorm 各 0.003 MiB）。扣掉参数，**单层真正"为反向保存的激活"≈ 276.5 MiB**。

用教科书的解析式对账（$B=4$, $S=512$, $H=12$, $d_k=64$, $d_{ff}=3072$, $d=768$，fp32）：

| 类别 | 内容 | 解析值 |
|---|---|---:|
| attention 的 $S^2$ 项 | `scores`（$B H S^2$ = 48.0 MiB）由 `where`→`softmax`,softmax 输出与其内部各留一份 = **2 份** | 96.0 MiB |
| attention 线性项 | Q/K/V 输出 + concat 后的 attn_output + output_proj 输出 | ~36 MiB |
| MLP 四项 | `w1(x)`、`sigmoid`、`silu` 乘积、`w3(x)`、最终乘积（$B S d_{ff}$ = 24.0 MiB/份） | ~96 MiB |
| residual / RMSNorm | 两个 sublayer 的残差加、两次 RMSNorm 输出（$B S d$ = 6.0 MiB/份） | ~30 MiB |
| **合计** | | **≈ 258–280 MiB** |

**解析 ≈ 276 MiB 与实测 276.5 MiB 吻合**，量级和构成都对上了。

### 结果 2：Top-5 贡献算子（占"为反向保存的全部内存"的百分比）

从 `small` ctx512 fp32 forward 快照里，取**在 forward 峰值时刻仍存活（即被反向用到）**的分配，按 `model.py` 调用点聚合：

| # | 调用点（栈回溯） | 语义 | MiB/block | 全模型合计 | **占残差总量** |
|---:|---|---|---:|---:|---:|
| 1 | `scaled_dot_product_attention()` @ `model.py:432` | `softmax(scores)` 的输出 + softmax 内部保存的张量（2 × $B H S^2$） | 96.2 | 1154.2 MiB | **44.1%** |
| 2 | `silu()` @ `model.py:532` | `x * torch.sigmoid(x)` 的 sigmoid 输出 + 乘积（2 × $B S d_{ff}$） | 48.0 | 576.0 MiB | **22.0%** |
| 3 | `forward()` @ `model.py:399` | SwiGLU 里的 `self.w3(x)`（$B S d_{ff}$） | 24.0 | 288.0 MiB | **11.0%** |
| 4 | `RMSNorm.forward()` @ `model.py:104` | `(self.weight * x).to(in_dtype)` 的输出（$B S d$） | 6.5 | 78.0 MiB | **3.0%** |
| 5 | `CausalMultiHeadSelfAttention.forward()` @ `model.py:524` | `rearrange(...).contiguous()` 拼头后的 attn_output（$B S d$） | 6.0 | 72.0 MiB | **2.8%** |
| | | **Top-5 合计** | **180.7** | **2168.2 MiB** | **82.9%** |

（百分比分母是**能归因到 `model.py` 的残差总量 2615.4 MiB**——占实测残差总量 3414.7 MiB 的 77%。剩下 23% 没有 `model.py` 栈帧：causal mask、`cross_entropy`、RotaryEmbedding 的 `cos/sin` 缓存、以及 PyTorch 内部的 autograd wrapper。注意 `model.py:427` 的 `einsum(Q,K)` 在**累计分配量**里排第二（1152.0 MiB），但在**残差集里消失了**——因为 `torch.where` 的反向不需要它的输入，scores 用完即释放。）

### 结果 3：反向产生的梯度张量 ≈ 36 MiB/block（+ ~36 MiB 临时量）

`full step` 斜率（384.2）− `forward-only` 斜率（312.6）= **71.6 MiB/block**，这才是反向在单层上净增的显存。拆开看：

| 项 | 计算 | 值 |
|---|---|---:|
| 参数梯度（常驻到 optimizer） | 9.4387 M × 4 B | **36.01 MiB** |
| 反向期间临时产生的激活梯度 | 71.6 − 36.0 | **≈ 35.6 MiB** |

### 回答

**单个 `TransformerBlock` 为反向保存的残差约 277 MiB**（边际峰值增量 312.6 MiB，扣掉该层自身 36.01 MiB 参数），**Top-5 算子吃掉了其中约 83%**：softmax 的 $B H S^2$ 中间量以 **44.1%** 独占鳌头，MLP 的 `silu` 与 `w3(x)` 合计 **33.0%**，两个 $B S d$ 量级的线性项只有约 4.9%——**残差开销几乎完全由 $S^2$ 项和 $d_{ff}$ 项决定，这正是 activation checkpointing 和 FlashAttention 的靶心**。

**反向的梯度显存与预期完全吻合**：full-step 与 forward-only 的斜率差为 **71.6 MiB/block**，其中 **36.01 MiB** 是参数梯度——**恰好等于该层参数本身（9.4387 M × 4 B）**，也就是"梯度张量与参数张量同尺寸"这条基本预期的直接验证；剩下的 **≈ 35.6 MiB** 是反向途中瞬时产生的激活梯度（主要是回传穿过 softmax 的那份 $B H S^2$ 量级张量，与 48 MiB 的 scores 同量级，还会被 allocator 与其它分配复用）。需要注意的是**这部分梯度并不常驻**：它随反向逐层向下传递、用完即释放，所以 `full step` 的峰值曲线在 backward 段整体是**下降**的——**只有参数梯度 + 优化器状态才是真正抬高峰值的那部分**。

### (f) 交付物说明：为什么改用 PyTorch memory snapshot

讲义要的是 Nsight Systems 的截图。**`nsys --cuda-memory-usage=true` 在本机是可以跑通的**（实测生成了 380 KB 报告）：

```bash
nsys profile --force-overwrite=true -o /tmp/memtest --cuda-memory-usage=true \
  --trace=cuda,nvtx -c cudaProfilerApi --capture-range-end=stop -- \
  .venv/bin/python -m cs336_systems.benchmark --model-size small --context-length 128 \
  --mode full --warmup 2 --steps 1 --nvtx --profiler-range
```

但**导出的数据回答不了 (f) 的问题**。`nsys stats` 导出 sqlite 后实测：

| 事实 | 实测值 |
|---|---|
| 分配事件表 `CUDA_GPU_MEMORY_USAGE_EVENTS` 行数 | 3855 |
| 其中 `memoryOperationType = Allocate` / `Deallocate` | **3855 / 0** |
| `start` 时间戳的**不同取值个数** | **1**（全部为 0，即**无时间信息**） |
| `localMemoryPoolSize` / `localMemoryPoolUtilizedSize` 非 NULL 行数 | **0 / 0** |
| 该表内分配总字节 | 2792.5 MB |
| `nsys stats` 自带的显存报表（`cuda_gpu_mem_time_sum` / `cuda_gpu_mem_size_sum`）覆盖范围 | **只有 memcpy / memset**，不含 `cudaMalloc` |

也就是说，nsys 在这个版本里给出的是一份**没有时间轴的分配清单**：既分不清哪次分配属于 forward、哪次属于 backward，也**完全没有释放记录**——而 (f) 问的恰恰是"反向把残差释放掉了多少、又新分配了多少梯度"。所以 nsys 这条路只能给出总量，给不出阶段分解。

因此改用**两条互相独立、可交叉验证的路**：

1. **PyTorch 官方 memory snapshot**（`torch.cuda.memory._dump_snapshot`）：`device_traces` 里 **`alloc` 与 `free_completed` 都带 `time_us`**，`frames` 字段还带**完整 Python 栈回溯**，能精确到 `model.py` 行号——比 nsys 的显存视图信息量更大。
2. **层数斜率实验**：直接测出"每 block 多少 MiB"，完全不依赖任何 profiler 的归因正确性。

两条路结论一致：残差量级重叠（帧归因 2615.4 MiB ⊆ 实测总量 3414.7 MiB），**参数梯度 36.01 MiB 精确对上**（斜率差 71.6 MiB − 参数梯度 36.0 MiB = 35.6 MiB，与解析值完全一致）。`profiles/memory/*.png` 即 (a)/(f) 的时间线交付物，`memory_report.py --top N` 的栈回溯即 (e)/(f) 的"操作归因"交付物。

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

# memory_profiling (a)(b)(c)：抓显存快照（每个配置一条命令，--steps 1 得到干净单步）
uv run python -m cs336_systems.benchmark --model-size small  --context-length 128  --mode forward --dtype fp32 --warmup 5 --steps 1 --memory-snapshot profiles/memory/small_ctx128_fp32_forward.pickle
uv run python -m cs336_systems.benchmark --model-size small  --context-length 128  --mode full    --dtype fp32 --warmup 5 --steps 1 --memory-snapshot profiles/memory/small_ctx128_fp32_full.pickle
uv run python -m cs336_systems.benchmark --model-size small  --context-length 512  --mode forward --dtype fp32 --warmup 5 --steps 1 --memory-snapshot profiles/memory/small_ctx512_fp32_forward.pickle
uv run python -m cs336_systems.benchmark --model-size medium --context-length 256  --mode full    --dtype fp32 --warmup 5 --steps 1 --memory-snapshot profiles/memory/medium_ctx256_fp32_full.pickle
# ... 其余 10 份同理（small/medium × ctx128/256/512/1024 × forward/full × fp32/bf16）

# memory_profiling (a)：把快照重建成 Active Memory Timeline（等价于 memory_viz，但可版本化）
uv run python -m cs336_systems.memory_report profiles/memory/small_ctx128_fp32_forward.pickle \
  --plot profiles/memory/small_ctx128_fp32_forward.png --title "small ctx128 fp32 forward"

# memory_profiling (e)(f)：打印最大分配的完整 Python 栈回溯
uv run python -m cs336_systems.memory_report profiles/memory/small_ctx512_fp32_forward.pickle --top 6

# memory_profiling (f)：层数斜率实验（单 block 残差 / 梯度显存）
#   small d768 ctx512 batch4 fp32，n_layers ∈ {6,9,12}，forward-only 斜率 312.6 MiB/block、
#   full-step 斜率 384.2 MiB/block，差值 71.6 MiB/block 即反向净增
uv run python -m cs336_systems.memory_per_block

# memory_profiling (f)：验证 nsys 的显存数据（结论：有分配清单、无时间轴、无释放记录）
nsys profile --force-overwrite=true -o /tmp/memtest --cuda-memory-usage=true \
  --trace=cuda,nvtx -c cudaProfilerApi --capture-range-end=stop -- \
  .venv/bin/python -m cs336_systems.benchmark --model-size small --context-length 128 \
  --mode full --warmup 2 --steps 1 --nvtx --profiler-range
nsys stats --force-export=true --report cuda_gpu_mem_size_sum /tmp/memtest.nsys-rep
python -c "import sqlite3;c=sqlite3.connect('/tmp/memtest.sqlite');\
print(list(c.execute('select count(distinct start), count(*) from CUDA_GPU_MEMORY_USAGE_EVENTS')))"
```
