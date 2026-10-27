# Profiling on the HUIT cluster: what works

Probed with `scripts/profiling_probe.sh` (Slurm job 4233, 2026-10-27) on one `gpu`-partition node:
**1x NVIDIA L4** (compute capability 8.9, 22 GiB, 58 SMs), driver 580.126.20, CUDA toolkit 13.0
(`nvcc` V13.0.88; torch 2.13.0+cu130). No model was loaded.

**Not probed:** the `gpu-parallel` (A10G) and `gpu-g7-reserved` nodes. The counter permission below is a
driver setting and can differ per node type; run the probe there before relying on it (`-p gpu-parallel
--gres=gpu:1`).

## Summary

| Tool | Works on the L4 node? | Notes |
|---|---|---|
| `nvidia-smi --query-gpu` fields | yes | `utilization.memory`, `clocks.sm`, `clocks.mem`, `clocks_event_reasons.active` (and the old `clocks_throttle_reasons.active` alias) all return values |
| `nvidia-smi dmon -s t` | yes | rxpci/txpci MB/s, one line per second, streams line by line through a pipe |
| DCGM (`dcgmi dmon`) | **no, as installed** | `dcgmi` 4.5.1 exists, but no host engine runs: `Unable to connect to host engine` (exit 235) |
| Nsight Compute `ncu` | **yes** | 2025.3.1; `RmProfilingAdminOnly: 0`; the matmul roofline below ran as a normal user |
| Nsight Systems `nsys` | **yes** | 2025.3.2; CUDA trace and kernel summary work; GPU-metrics sampling works |

Tools are installed but **not on `PATH`**: `ncu`, `nsys`, `nvcc` are in `/usr/local/cuda/bin` (-> `cuda-13.0`),
`dcgmi` and `nv-hostengine` in `/usr/bin`. Add `export PATH=/usr/local/cuda/bin:$PATH` in the job.

## Permission (the usual ncu blocker)

```
$ grep -i RmProfilingAdminOnly /proc/driver/nvidia/params
RmProfilingAdminOnly: 0
$ sudo -n true      # no passwordless sudo (not needed, and nothing was changed)
```
`0` means non-admins may read performance counters, and `ncu` and `nsys --gpu-metrics-devices` ran as the
normal user. Had it been `1` the error is `ERR_NVGPUCTRPERM` and only a node admin can fix it (no workaround
on the cluster; ask HUIT research computing, use a machine with root, or fall back to nsys timelines plus
analytic FLOP/byte accounting). **Check this line on every new node type.**

## nvidia-smi (what the `gpu_metrics = "detailed"` sampler uses)

All probed fields work; `nvidia-smi --help-query-gpu` lists both `clocks_event_reasons.*` (new) and
`clocks_throttle_reasons.*` (old), and the sampler prefers the new name. Idle vs busy on the same GPU
(`ncu_smoke.py sample`, 5 s idle then 8 s of 4096x4096 bf16 matmuls):

| key | idle | busy |
|---|---|---|
| `gpu0_util_pct` | 0 | 91.3 |
| `gpu0_mem_util_pct` | 0 | 7.2 |
| `gpu0_clock_sm_mhz` | 210 | 1043 |
| `gpu0_clock_mem_mhz` | 405 | 5973 |
| `gpu0_power_w` | 16.2 | 65.8 |
| `gpu0_throttle_frac` | 0 | 0.90 (mask `0x4` = SW power cap; the L4 is a 72 W card) |
| `gpu0_pcie_tx_mb_s` | 0 | 6.6 |
| `gpu_util_pct_spread` | 0 | 0 (one GPU; it needs a multi-GPU node to be non-zero) |

The throttle result is the useful surprise: a busy L4 runs power-capped, which pooled utilization hides.

## DCGM

`dcgmi discovery -l` and `dcgmi dmon -e 1002,1003,1004,1005,1009,1010 -c 5 -d 500` both fail with
`Error: Unable to connect to host engine` because nothing runs `nv-hostengine` on the compute node. We did
**not** try starting one (that is a daemon; ask first). So the Stage-3 DCGM sampler was not built. The
same quantities are available from the nsys GPU-metrics sampler below.

## Nsight Systems

```bash
export PATH=/usr/local/cuda/bin:$PATH
nsys profile --trace=cuda,nvtx --stats=true -o out/nsys_basic python script.py        # timeline + summaries
nsys profile --trace=cuda --gpu-metrics-devices=cuda-visible -o out/nsys_gm python script.py
```
- CUDA API and kernel summaries work (`cuda_api_sum`, `cuda_gpu_kern_sum`). NVTX is empty unless the
  script emits ranges.
- `--gpu-metrics-device=all` is deprecated; use `--gpu-metrics-devices=cuda-visible` (or `all`).
- GPU-metrics sampling works at ~10 kHz. `nsys stats --report gpu_metrics` does **not** exist in this
  version; read the `GPU_METRICS` and `TARGET_INFO_GPU_METRICS` tables of the exported `.sqlite`
  (`metricName`, `value`). Metrics seen on the L4: SMs Active, SM Issue, Tensor Active, Compute Warps in
  Flight, DRAM Read/Write Bandwidth, PCIe RX/TX Throughput, GR Active, clocks. In the matmul run Tensor
  Active peaked at 96%.

## Nsight Compute: roofline method validated on bf16 matmuls

`scripts/ncu_smoke.py run` does, per size, fresh operands, a warm-up matmul, a `spin` marker kernel and the
measured matmul. The parser finds the marker in ncu's CSV and takes the GEMM kernels after it.

```bash
export PATH=/usr/local/cuda/bin:$PATH
ncu --target-processes all --csv \
  --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum,dram__bytes_write.sum,sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active \
  --log-file ncu.csv python scripts/ncu_smoke.py run --sizes 512,1024,2048,4096
python scripts/ncu_smoke.py parse ncu.csv --gpu "NVIDIA L4"
```
Metric names are architecture specific: the tensor-pipe metric above exists on this Ada (sm_89) chip;
`ncu --query-metrics | grep -i tensor` lists the alternatives (newer `_v2` instruction counters and
`sm__ops_path_tensor_src_bf16_dst_fp32`). The CSV `Metric Unit` for durations is `ns`.

Result (L4; spec peaks 121 dense bf16 TFLOP/s, 300 GB/s, ridge 403 FLOP/B):

| n | kernel | us | DRAM MB | TFLOP/s | % peak | AI FLOP/B | ideal AI (n/3) | sm% | mem% | tensor% |
|---|---|---|---|---|---|---|---|---|---|---|
| 512 | ampere_bf16_s16816gemm 128x64 | 19.6 | 1.27 | 13.7 | 11 | 211 | 171 | 29 | 28 | 61 |
| 1024 | ampere_bf16_s1688gemm 128x128 | 81.1 | 5.03 | 26.5 | 22 | 427 | 341 | 56 | 45 | 67 |
| 2048 | ampere_bf16_s1688gemm 128x128 | 425.3 | 20.20 | 40.4 | 33 | 850 | 683 | 86 | 56 | 88 |
| 4096 | ampere_bf16_s1688gemm 128x128 | 3123.9 | 164.64 | 44.0 | 36 | 835 | 1365 | 93 | 55 | 95 |

Reading it: SM and tensor-pipe utilization climb with size (29% -> 93% SM, 61% -> 95% tensor), arithmetic
intensity is above the ridge from n=1024 up (compute-bound), and n=512 sits below it with lower
utilization. Measured AI is below the ideal at n=4096 because tiled kernels re-read operands.

Caveats on the numbers:
- **ncu locks clocks to base by default**, so its TFLOP/s is below the unprofiled run. CUDA-event timing
  without a profiler gave 25.2 / 57.2 / 72.5 / 57.1 TFLOP/s for the same sizes. Use `--clock-control none`
  to measure at boost clocks. Use ncu for ratios and bytes, not absolute speed.
- The L4 is power-capped (72 W), which also lowers sustained clocks (see the throttle row above).
- Peaks come from spec sheets in `PEAKS` in `ncu_smoke.py`; check them for other GPUs.
- ncu caches are flushed between replays by default, so DRAM bytes are cold-cache numbers.

## What this means for model profiling

- **ncu replays each kernel** several times (one pass per counter group), so it is practical only for a
  handful of kernels chosen after an nsys run finds the top ones. It is not for a full sweep.
- **vLLM tensor-parallel workers are separate processes.** ncu needs `--target-processes all` and is clumsy
  with TP=4 workers; for nsys also set `VLLM_WORKER_MULTIPROC_METHOD=spawn`,
  `--trace-fork-before-exec=true` and `--cuda-graph-trace=node`.
- **The in-process PyTorch models** (SemIf, Kev, Laya, Bongard, decider) are the better ncu targets: one
  process, one GPU, and `torch.cuda.nvtx` ranges or marker kernels delimit the region.
- A sweep that needs SM, tensor and DRAM activity over time can use the nsys GPU-metrics sampler instead of
  DCGM until a host engine is available.

## Rerunning the probe

```bash
mkdir -p ~/probes/logs ~/probes/pkg/s1 && touch ~/probes/pkg/s1/__init__.py
cp scripts/profiling_probe.sh scripts/ncu_smoke.py ~/probes/
cp src/s1/gpu_stats.py src/s1/record.py ~/probes/pkg/s1/
cd ~/probes && sbatch -p gpu --gres=gpu:1 -c 4 --mem=16G -t 00:10:00 -J profiling-probe \
  -o ~/probes/logs/%j.out ~/probes/profiling_probe.sh
```
Run `sbatch` from the directory holding `ncu_smoke.py`: Slurm runs a copy of the script from its spool
directory. Raw ncu/nsys files land in `~/probes/out/<jobid>/`.
