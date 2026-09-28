# nsys_profile 结果汇总（自动生成）

## ① forward-only vs 完整训练步：matmul 占比（讲义 (d)）

| 模型 | context | forward matmul% | full matmul% | 变化 | nsys measure/步 fwd (ms) | timeit fwd (ms) | nsys measure/步 full (ms) | timeit full (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| medium | 128 | 84.6% | 39.5% | -45.0 pt | 49.7 | 51.0 | 296.0 | 294.7 |
| medium | 256 | 74.9% | 47.6% | -27.3 pt | 103.0 | 104.0 | 461.5 | 462.7 |
| medium | 512 | 58.4% | 45.2% | -13.2 pt | 265.3 | 266.5 | 965.1 | 967.0 |
| small | 1024 | 41.1% | 35.3% | -5.9 pt | 240.4 | 240.6 | 818.2 | 819.5 |
| small | 256 | 72.5% | 45.7% | -26.8 pt | 31.9 | 31.9 | 144.9 | 143.9 |
| small | 512 | 53.0% | 41.3% | -11.7 pt | 86.8 | 87.0 | 316.3 | 316.6 |

> `nsys measure/步` = `nvtx_gpu_proj_sum` 里 `:measure` 的投影 GPU 时间 ÷ 3 步；
> 和 timeit 的单步均值应当接近（若差得远，说明 profiling 本身的开销不可忽略）。

## ② 逐份 profile 明细

### `medium_ctx128_forward`

- **matmul 时间占比：84.6%**（其余 15.4%）

| Time % | Instances | Kernel |
|---:|---:|---|
| 59.2% | 216 | `ampere_sgemm_128x64_tn` |
| 23.2% | 291 | `ampere_sgemm_64x64_tn` |
| 3.1% | 144 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::...` |
| 2.9% | 870 | `void at::native::elementwise_kernel<(int)128, (int)2, void at::nati...` |
| 1.8% | 432 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::...` |
| 1.3% | 72 | `ampere_sgemm_128x128_nn` |
| 1.0% | 144 | `void at::native::elementwise_kernel<(int)128, (int)2, void at::nati...` |
| 0.9% | 144 | `void at::native::<unnamed>::CatArrayBatchedCopy<at::native::<unname...` |

### `medium_ctx128_full`

- **matmul 时间占比：39.5%**（其余 60.5%）

| Time % | Instances | Kernel |
|---:|---:|---|
| 12.5% | 507 | `ampere_sgemm_64x64_nt` |
| 9.7% | 216 | `ampere_sgemm_128x64_tn` |
| 9.2% | 126 | `void at::native::<unnamed>::multi_tensor_apply_kernel<at::native::<...` |
| 9.1% | 219 | `ampere_sgemm_128x64_nn` |
| 8.6% | 63 | `void at::native::<unnamed>::multi_tensor_apply_kernel<at::native::<...` |
| 6.7% | 63 | `void at::native::<unnamed>::multi_tensor_apply_kernel<at::native::<...` |
| 6.7% | 63 | `void at::native::<unnamed>::multi_tensor_apply_kernel<at::native::<...` |
| 5.5% | 1083 | `void at::native::elementwise_kernel<(int)128, (int)2, void at::nati...` |

### `medium_ctx256_forward`

- **matmul 时间占比：74.9%**（其余 25.1%）

| Time % | Instances | Kernel |
|---:|---:|---|
| 69.9% | 504 | `ampere_sgemm_128x64_tn` |
| 6.8% | 144 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::...` |
| 2.9% | 870 | `void at::native::elementwise_kernel<(int)128, (int)2, void at::nati...` |
| 2.8% | 72 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::...` |
| 2.2% | 72 | `ampere_sgemm_128x128_nn` |
| 1.8% | 432 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::...` |
| 1.8% | 72 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::...` |
| 1.7% | 3 | `ampere_sgemm_64x64_tn` |

### `medium_ctx256_full`

- **matmul 时间占比：47.6%**（其余 52.4%）

| Time % | Instances | Kernel |
|---:|---:|---|
| 15.1% | 504 | `ampere_sgemm_128x64_tn` |
| 14.9% | 507 | `ampere_sgemm_64x64_nt` |
| 14.7% | 507 | `ampere_sgemm_128x64_nn` |
| 7.7% | 1164 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::...` |
| 5.9% | 126 | `void at::native::<unnamed>::multi_tensor_apply_kernel<at::native::<...` |
| 5.5% | 63 | `void at::native::<unnamed>::multi_tensor_apply_kernel<at::native::<...` |
| 4.3% | 63 | `void at::native::<unnamed>::multi_tensor_apply_kernel<at::native::<...` |
| 4.3% | 63 | `void at::native::<unnamed>::multi_tensor_apply_kernel<at::native::<...` |

### `medium_ctx512_forward`

- **matmul 时间占比：58.4%**（其余 41.6%）

| Time % | Instances | Kernel |
|---:|---:|---|
| 26.7% | 147 | `ampere_sgemm_64x64_tn` |
| 25.6% | 360 | `ampere_sgemm_128x64_tn` |
| 7.2% | 144 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::...` |
| 4.9% | 72 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::...` |
| 4.8% | 72 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::...` |
| 4.0% | 72 | `void at::native::elementwise_kernel<(int)128, (int)2, void at::nati...` |
| 4.0% | 72 | `void at::native::elementwise_kernel<(int)128, (int)2, void at::nati...` |
| 3.6% | 72 | `void at::native::elementwise_kernel<(int)128, (int)2, void at::nati...` |

### `medium_ctx512_full`

- **matmul 时间占比：45.2%**（其余 54.8%）

| Time % | Instances | Kernel |
|---:|---:|---|
| 13.9% | 507 | `ampere_sgemm_64x64_nn` |
| 10.9% | 1164 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::...` |
| 8.5% | 219 | `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x128_8x4_nt_align1>(...` |
| 7.3% | 147 | `ampere_sgemm_64x64_tn` |
| 7.0% | 360 | `ampere_sgemm_128x64_tn` |
| 5.1% | 288 | `void at::native::elementwise_kernel<(int)128, (int)2, void at::nati...` |
| 4.3% | 1371 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::...` |
| 3.4% | 288 | `ampere_sgemm_64x64_nt` |

### `small_ctx1024_forward`

- **matmul 时间占比：41.1%**（其余 58.9%）

| Time % | Instances | Kernel |
|---:|---:|---|
| 31.9% | 255 | `ampere_sgemm_128x64_tn` |
| 8.1% | 36 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::...` |
| 8.0% | 36 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::...` |
| 7.6% | 36 | `void at::native::elementwise_kernel<(int)128, (int)2, void at::nati...` |
| 7.6% | 36 | `void at::native::elementwise_kernel<(int)128, (int)2, void at::nati...` |
| 5.9% | 36 | `void at::native::elementwise_kernel<(int)128, (int)2, void at::nati...` |
| 5.9% | 72 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::...` |
| 5.0% | 36 | `ampere_sgemm_128x128_nn` |

### `small_ctx1024_full`

- **matmul 时间占比：35.3%**（其余 64.7%）

| Time % | Instances | Kernel |
|---:|---:|---|
| 13.3% | 588 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::...` |
| 9.4% | 255 | `ampere_sgemm_128x64_tn` |
| 9.3% | 144 | `void at::native::elementwise_kernel<(int)128, (int)2, void at::nati...` |
| 9.0% | 252 | `ampere_sgemm_64x64_nn` |
| 6.0% | 687 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::...` |
| 5.8% | 111 | `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x128_8x4_nt_align1>(...` |
| 4.7% | 72 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::...` |
| 4.7% | 72 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::...` |

### `small_ctx256_forward`

- **matmul 时间占比：72.5%**（其余 27.5%）

| Time % | Instances | Kernel |
|---:|---:|---|
| 37.7% | 183 | `ampere_sgemm_64x64_tn` |
| 30.6% | 72 | `ampere_sgemm_128x64_tn` |
| 5.8% | 72 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::...` |
| 3.5% | 438 | `void at::native::elementwise_kernel<(int)128, (int)2, void at::nati...` |
| 3.1% | 36 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::...` |
| 2.6% | 36 | `ampere_sgemm_128x128_nn` |
| 2.2% | 216 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::...` |
| 2.1% | 36 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::...` |

### `small_ctx256_full`

- **matmul 时间占比：45.7%**（其余 54.3%）

| Time % | Instances | Kernel |
|---:|---:|---|
| 13.3% | 252 | `ampere_sgemm_64x64_nt` |
| 11.1% | 219 | `ampere_sgemm_64x64_nn` |
| 8.1% | 588 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::...` |
| 8.0% | 183 | `ampere_sgemm_64x64_tn` |
| 6.5% | 72 | `ampere_sgemm_128x64_tn` |
| 5.8% | 42 | `void at::native::<unnamed>::multi_tensor_apply_kernel<at::native::<...` |
| 5.4% | 21 | `void at::native::<unnamed>::multi_tensor_apply_kernel<at::native::<...` |
| 4.2% | 21 | `void at::native::<unnamed>::multi_tensor_apply_kernel<at::native::<...` |

### `small_ctx512_forward`

- **matmul 时间占比：53.0%**（其余 47.0%）

| Time % | Instances | Kernel |
|---:|---:|---|
| 24.7% | 75 | `ampere_sgemm_64x64_tn` |
| 21.4% | 180 | `ampere_sgemm_128x64_tn` |
| 8.1% | 72 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::...` |
| 5.6% | 36 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::...` |
| 5.5% | 36 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::...` |
| 4.4% | 36 | `void at::native::elementwise_kernel<(int)128, (int)2, void at::nati...` |
| 4.3% | 36 | `void at::native::elementwise_kernel<(int)128, (int)2, void at::nati...` |
| 4.1% | 36 | `void at::native::elementwise_kernel<(int)128, (int)2, void at::nati...` |

### `small_ctx512_full`

- **matmul 时间占比：41.3%**（其余 58.7%）

| Time % | Instances | Kernel |
|---:|---:|---|
| 12.3% | 255 | `ampere_sgemm_64x64_nn` |
| 12.0% | 588 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::...` |
| 7.8% | 111 | `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x128_8x4_nt_align1>(...` |
| 6.7% | 75 | `ampere_sgemm_64x64_tn` |
| 5.8% | 180 | `ampere_sgemm_128x64_tn` |
| 5.7% | 144 | `void at::native::elementwise_kernel<(int)128, (int)2, void at::nati...` |
| 4.7% | 687 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::...` |
| 3.1% | 72 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::...` |

---

## ⚠️ NVTX 投影的两个已知陷阱（本机实测）

1. **`:backward` 的投影≈0，不可信。** PyTorch 的 autograd 引擎在**独立的 device 线程**上
   发射 kernel，主线程上 `torch.cuda.nvtx.range("backward")` 推的区间盖不住它们
   （实测：backward 只归因到 1 个 kernel，而 34 种 kernel / 936ms 完全未被任何区间覆盖）。
   要覆盖 backward 得加 `--pytorch=autograd-nvtx`，但那样区间会细到**每个算子一个**
   （`aten::_foreach_addcdiv_, op_id=18667`），反而不好聚合。
2. **`:optimizer` 的投影被高估。** `optimizer.step()` 在 CPU 上紧接着 `backward()` 返回就执行，
   而 backward 的 GPU kernel 还在飞——于是**反向的尾巴被算进了 optimizer 的窗口**
   （实测 medium_ctx128 的 `:optimizer` 显示 395ms，比整个 forward 的 147ms 还大，不可能）。

**结论：只有两个数字可信** ——
`：measure`（整步的 GPU 时间）与 `:forward`（每步第一段，前面没有异步尾巴）。
**backward + optimizer 的量请用「forward-only 与 full 两份 profile 的差值」得到**，
这正是上面 ① 表的用法。