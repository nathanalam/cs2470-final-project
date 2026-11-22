# Picking the steady-state batch automatically: design (2026-11-22)

Status: **implemented** in `src/s1/steady_state.py` (usage and the table: `docs/steady_state_batch.md`), with
the simple formula and the table-decides default policy. Not implemented yet from section 5: the
halve-once fallback on OOM, recording vLLM's `max_num_batched_tokens`, the report's "picked by" column and
bracket drift check, and the `vl_batch` hook (that benchmark is on the multimodal branch). No GPU or Modal
time was spent. The numbers below come from the
committed records (`results/*.json`), `results/steady_state_findings.md`, `docs/multimodal.md` on
`claude/multimodal-exp-53e3b0`, and a web literature pass (sources are cited inline).

The method we agreed on 2026-11-19 is to compare every model x GPU pair at its own steady state. That means GPU
kernel time >= 95% of wall time, outside the overhead-bound batch-1 region. Batch size is only the means of
getting there. This doc designs code that picks that batch (or, for vLLM, the concurrency) for each run. It also
checks the measured B\* values against theory and outside measurements, and proposes defaults for pairs we have
not swept.

## 1. Summary

- **Recommendation: the pinned table decides; a formula is an opt-in fallback; no online calibration.**
  - The table (`src/s1/steady_state.py`) holds every measured B\*, keyed by model, *variant* (host prep, expert
    dispatch, mode), GPU and state length, together with the run IDs it came from. A test recomputes each entry
    from those records.
  - **The table always wins when it has an entry.** By default (`steady_policy = "table"`), a pair that is not
    in the table fails fast before the model loads, with a message naming the sweep to run. The formula never
    overrides a table entry.
  - Only a config that sets `steady_policy = "table+formula"` lets an unswept pair run on the formula: a
    one-line estimate from the GPU, the model's active parameter count and the request length. Its inputs and
    result go into the record, marked `predicted`. This is for exploratory runs on new GPUs; a result worth
    citing should come from a table entry.
  - An opt-in `"steady-bracket"` mode runs {B/2, B, 2B} so that a run can check its own pick.
- **The formula** (tokens per forward needed, then batch):

  ```
  tokens = T[gpu] * max(1, 8B / active_params)
  B      = clamp( 2 ** round(log2(tokens / row_tokens(L))),  1,  B_mem(L) )
  ```

  | Symbol | Meaning |
  |---|---|
  | `T[gpu]` | Tokens per forward that keep this GPU busy for an ~8B dense model: L4 1.5k, A100 3k, H100/H200 6k, B200 12k (section 4.1) |
  | `active_params` | Parameters used per token (`flops.Arch.active_params`); for an MoE, the active count, not the total |
  | `row_tokens(L)` | Tokens in one request: state length plus ~270 for the JEVal question and options |
  | `B_mem` | The memory cap (section 3.5) |

  **In plain terms.** Every forward pass carries a fixed CPU cost of about 10-30 ms to launch it, whatever the
  model. The GPU only stays 95% busy if it has much more work than that per forward. A big model on a slow GPU
  already has enough work in a single request; a small model on a fast GPU needs many requests at once. So start
  from the GPU's number in the table. If the model is smaller than ~8B active parameters, scale that number up
  in proportion, because each token gives the GPU less work. Divide by how long one request is, and round to a
  power of two.

  **Back-test.** Applied to all 36 measured (model, variant, GPU, length) series, this pick reaches >= 95% of
  the throughput at the measured B\* (or the plateau, when B\* was not reached) in **31 of 36**. The worst case
  is 82%. A more detailed version that models host time from kernel counts (section 4.5) scores exactly the
  same, so the design uses the simple one.
- **Main finding from theory and data: "tokens in flight" depends on the model, not only the GPU.**
  - Dense 8-9B models saturate at about 1.5k tokens per forward on the L4 and 2-7k on Hopper.
  - LFM2-8B-A1B (1.5B active, host-sync-heavy MoE) needs **21-26k** on Hopper.
  - The cause is that host time per forward scales with kernel count, while GPU time scales with active FLOPs.
    A per-GPU T_target alone (the parent design's B\* ≈ ceil(T_target / state_tokens)) is off by about 5x for
    small-active models.
- **The research mostly agrees with our numbers.** Ridges of ~160-420 FLOP/byte mean every point we call
  steady is deeply compute-bound. The H200 ≈ H100 + 4% result fits. Engine token budgets (vLLM 2k/8k/16k by
  GPU tier, TRT-LLM 8k, TGI 4k) bracket our dense M\*. Eager MoE and inline tokenization behave as the sources
  predict.
- **Two disagreements:**
  1. The research subagent blamed the L4 OOM on full-sequence logits. SemIf already passes `logits_to_keep=1`
     (`semif.py:199`), so the cause is elsewhere, and the `batch` benchmark does not record peak memory at all.
     Adding it is part of this design.
  2. A GPU-class-only T_target overshoots badly for dense models at 1024 tokens on the L4. Throughput *falls*
     beyond B\* there (0.76-0.80x at B = 8), so overshooting is not free.
- **Correction to the brief.** Kev-4B and Laya *were* swept, in the 2026-11-09 reference sweep
  (`configs/batch_sweep_ref.toml`):
  - Kev-4B reaches B\* only on the L4 (8 at 128, 1 at 1024) and the A100 (16 at 128).
  - Laya never reaches 0.95 in any mode on any GPU.

## 2. What we measured (recomputed from the records)

I ran `report_steady.steady` over every `batch` series (`report_batch.panels`). `tok@B*` is the padded token
count of one forward at the reported batch, from the record's `batch_s<L>_b<B>_tokens`. A row is about L + 270
tokens, because the question and options add ~270 tokens to the reference state length.

| GPU | Model (variant) | L | B\* | Plateau | tok@B\* (or plateau) | Max kernel_busy | Run |
|---|---|---|---|---|---|---|---|
| L4 | qwen3-8b (eager) | 128 / 1024 | 4 / 1 | 2 / 1 | 2.3k / 1.3k | 1.0+ | 20261109-231713-a38ec98c |
| L4 | qwen35-9b | 128 / 1024 | 4 / 1 | 2 / 1 | 2.3k / 1.3k | 1.0+ | 20261119-225425-e0d0badf |
| L4 | jevk5-9b | 128 / 1024 | 2 / 1 | 2 / 1 | 0.9k / 1.3k | 1.0+ | 20261119-230530-ed7206a6 |
| L4 | lfm2-8b-a1b (inline) | 128 / 1024 | – / – | 2 / 1 | 1.0k / 1.3k | 0.94 | 20261119-224843-9162d873 |
| L4 | kev-4b (fused + graphs) | 128 / 1024 | 8 / 1 | 8 / 1 | 2.7k / 1.2k | 1.0+ | 20261109-230438-4ddb0b2d |
| A100-SXM4 | qwen3-8b | 128 / 1024 | 1 / 1 | 1 / 1 | 0.4k / 1.3k | 1.0+ | 20261109-231247-cc3371f7 |
| A100-SXM4 | kev-4b | 128 / 1024 | 16 / – | 16 / 2 | 5.3k / 2.4k | 0.95 / 0.86 | 20261109-230438-7a7d5a81 |
| A100-SXM4 | lfm2-8b-a1b (inline) | 128 / 1024 | – / – | 32 / 32 | 21k / 51k | 0.93 / 0.91 | 20261119-222604-2a4972b9 |
| H100 | qwen3-8b | 128 / 1024 | 4 / 4 | 4 / 1 | 2.3k / 5.9k | 0.96 / 0.97 | 20261109-211259-b63a8b3e |
| H100 | lfm2-8b-a1b (pretokenized) | 128 / 1024 | **32 / 16** | 32 / 16 | **21k / 26k** | 0.97 / 0.98 | 20261119-230040-acb6fcff |
| H100 | lfm2-8b-a1b (inline) | 128 / 1024 | – / – | 32 / 16 | 21k / 26k | 0.87 / 0.86 | 20261119-222604-5a3f7682 |
| H100 | lfm2-8b-a1b (eager experts) | 128 / 1024 | – / – | 64+ | 42k+ / 101k+ | 0.71 / 0.88 | 20261119-225421-13a334e9, -230224-00670377 |
| H100 | kev-4b | 128 / 1024 | – / – | 16 / 2 | 5.3k / 2.4k | 0.89 / 0.27 | 20261109-205016-9d8b32e6 |
| H200 | qwen3-8b | 128 / 1024 | 4 / 4 | 2 / 1 | 2.3k / 5.9k | 0.97 | 20261109-231248-7eb18713 |
| H200 | lfm2-8b-a1b (inline) | 128 / 1024 | – / – | 32 / 16 | 21k / 26k | 0.87 | 20261119-222601-ced5a93f |
| B200 | qwen3-8b | 128 / 1024 | – / 4 | 64 / 1 | 42k / 5.9k | 0.96 / 1.04 | 20261109-231135-fd87886c |
| B200 | lfm2-8b-a1b (inline) | 128 / 1024 | – / – | 32 / 16 | 21k / 26k | 0.92 / 0.94 | 20261119-222600-33e72d92 |
| B200 / H200 / H100 | kev-4b | 128 / 1024 | – / – | 16 / 2-8 | 5.3k / 2.4-9.7k | 0.86-0.97 / 0.23-0.32 | 11-09 records |
| all GPUs | laya (4 modes) | both | – | 4-64 | – | ≤ 0.92 | 11-09 records |

**d1-3B** (`vl_batch`, `docs/multimodal.md`, runs `20261120-141226-e1dd7b35` L4 and `-141037-d11b3b27` H100):

| Arm | L4 B\* | ≈ tokens | H100 B\* | ≈ tokens | Note |
|---|---|---|---|---|---|
| Text rows (~60 tokens) | 32 | 1.9k | 64 | 3.8k | |
| Text at 384 px length | 16 | 2.5k | 64 | 10k | |
| Text at 1024 px length (1.8k tokens) | 1 | 1.8k | 4 | 7.4k | |
| 1024 px images | 1 | | – | | Never reaches 0.95 on H100 (0.87-0.91; fastest B = 32) |

**OpenJev 27B on vLLM** (`jeval_concurrency`, 1,000 questions): steady at **8 on all four GPUs**.

| GPU | Run | Per-GPU util at 1 | Util at 8 | q/s at 8 | Gain from 8 to 128 |
|---|---|---|---|---|---|
| A100 80GB PCIe | 20261119-224354-d73f77fb | 93% | 99.7% | 2.22 | ≤ 3.5% |
| H100 | 20261119-223636-995400ee | 70% | 98-99% | 6.77 | ≤ 3.5% |
| H200 | 20261119-223601-41b4cce3 | 68% | 98-99% | 7.03 | ≤ 3.5% |
| B200 | 20261119-223618-6c6bc83f | 64% | 98-99% | 12.60 | ≤ 3.5% |

**Two patterns in this data:**
1. A dense model at 1024 tokens is usually steady at B = 1-4, because one row already carries ~1.3k tokens.
2. Every series that never reaches 0.95 fails for a host reason that batching cannot fix:
   - LFM2 inline tokenization;
   - eager expert loops;
   - Laya's per-row encoder work;
   - the unexplained Kev-4B 1024-token floor on Hopper/Blackwell;
   - d1 image preprocessing and the unpinned `pixel_values` copy.

## 3. Research and how it compares

The full report (all URLs) came from a web-research subagent. The parts the design relies on are below.

### 3.1 Theory: when a prefill forward stops being overhead-bound

There are two separate thresholds.

**(a) GEMM shape.** A bf16 linear layer with M = B x tokens rows has arithmetic intensity ≈ M FLOP/byte while
M ≪ K, N. It turns compute-bound at M ≈ ridge = peak FLOP/s / HBM bandwidth.
- kipply's A100 example gives 208 (https://kipp.ly/transformer-inference-arithmetic/).
- DistServe: a 512-token prefill of a 13B model already makes an A100 "near compute-bound", after which more
  requests no longer improve efficiency (https://arxiv.org/html/2401.09670).
- SARATHI: Llama-13B prefill peaks at ≥ 512 tokens on an A6000; larger hidden sizes saturate with fewer tokens
  (https://arxiv.org/pdf/2308.16369).

Ridges are computed from datasheet dense bf16 FLOP/s and HBM bandwidth:

| GPU | Dense bf16 TFLOP/s | Bandwidth | Ridge (FLOP/B) | Our measured bf16 peak (median of runs) | Source |
|---|---|---|---|---|---|
| L4 | 121 | 300 GB/s | ~400 | 56 | https://www.nvidia.com/en-us/data-center/l4/ |
| A10 (A10G proxy) | 125 | 600 GB/s | ~210 | – | https://www.nvidia.com/en-us/data-center/products/a10-gpu/ |
| A100 80GB | 312 | 1.9-2.0 TB/s | ~160 | 230 (PCIe), 259 (SXM4) | https://www.nvidia.com/en-us/data-center/a100/ |
| L40S | 362 | 864 GB/s | ~420 | 186 | https://www.nvidia.com/en-us/data-center/l40s/ |
| H100 SXM | ~989 | 3.35 TB/s | ~295 | 674 | https://www.nvidia.com/en-us/data-center/h100/ |
| H200 | ~989 | 4.8 TB/s | ~205 | 676 | https://www.nvidia.com/en-us/data-center/h200/ |
| B200 | ~2,250 | ~8 TB/s | ~280 | 1,340 | secondary: https://inferencex.semianalysis.com/chips/b200 |
| RTX Pro 4500 Blackwell | 406 (basis unclear) | 800-896 GB/s | ~250-510 | – | https://www.nvidia.com/en-us/data-center/rtx-pro-4500-blackwell-server-edition/ |

Caveats: AWS publishes no bf16 figure for the A10G, only "250 TOPS", and no NVIDIA-primary B200 per-GPU sheet
was reachable. **Agreement:** all of our steady-state points sit at 1.6k-12k FLOP/byte (roofline records,
multimodal doc), far right of every ridge. So the GEMM threshold (a) is reached at a few hundred tokens and
does not explain B\* in the thousands.

**(b) Host overhead.** This is what sets B\* for us.
- A kernel launch costs a few µs (NVIDIA: 3.8 µs back-to-back on V100, https://developer.nvidia.com/blog/cuda-graphs/).
- Eager PyTorch adds Python dispatch. Horace He measured ~280k tiny ops/s. His diagnostic is our B\* test: "if
  you double your batch size but your runtime only increases by 10%, you're likely overhead bound"
  (https://horace.io/brrr_intro.html).
- Our records measure this directly: `launch_api_ms / kernels` at B = 1 is **2.2-8 µs per kernel** on Modal
  hosts (Zen 5 fastest, Skylake slowest). That gives 10-30 ms of host time per eager forward for ~1.5-3k
  kernels.

If that host time does not overlap the GPU (host syncs: MoE routing, `.cpu()` readout, tokenization inside the
timed region), kernel_busy ≈ G / (G + H). Reaching 0.95 then needs GPU time G ≥ 19·H, so:

```
M_host = 19 * H * F_eff / F_tok
```

with F_tok the FLOPs per token, roughly 2 x active parameters plus attention. This is why small-active models
need many more tokens:
- **LFM2:** F_tok = 2.9 GFLOP, H ≈ 10 ms on an H100, F_eff ≈ 400 TFLOP/s → 26k tokens. Measured: 21k (B\* 32 at
  128) and 26k (B\* 16 at 1024).
- **Qwen3-8B:** F_tok = 14 GFLOP, H ≈ 13.5 ms → 7.4k tokens. Measured: 2.3-5.9k.

The vLLM hybrid-model post makes the same point qualitatively. Triton launch overhead "can severely affect
performance … for small batches and/or models with a small number [of] activated parameters"
(https://pytorch.org/blog/hybrid-models-as-first-class-citizens-in-vllm/).

**Verdict on B\* ≈ ceil(T_target / state_tokens).**
- **Sound in form**, with two changes:
  - T_target must depend on the model as well as the GPU. Because H is about the same (10-30 ms) for every
    model we run, M_host mostly varies with F_tok, i.e. with active parameters. That is what the simple
    formula's `max(1, 8B / active_params)` factor captures.
  - The denominator must be the *row* tokens (state plus question, ~L + 270 for JEVal), padded.
- **Ceil is wrong; round in log2 instead.** Dense 9B models on the L4 at 1024 tokens lose 20-24% throughput at
  B = 8 versus B = 1, so overshooting has a cost.

### 3.2 Practice: what serving engines default to

vLLM V1 `get_batch_defaults`, read from `main`
(https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/engine/arg_utils.py). Verify these against the
pinned vLLM version before relying on them.

| Tier | max_num_batched_tokens (LLM class / API server) | max_num_seqs |
|---|---|---|
| ≥ 160 GiB (B200) | 16384 / 16384 | 1024 |
| ≥ 70 GiB, not A100 (H100, H200) | 16384 / 8192 | 1024 |
| Everything else (A100, L4, A10G, L40S) | 8192 / 2048 | 256 |

- vLLM's docs recommend > 8192 "especially for smaller models on large GPUs"
  (https://docs.vllm.ai/en/latest/configuration/optimization.html).
- TensorRT-LLM defaults `max_num_tokens` to 8192 and suggests sweeping powers of two ≥ 1024
  (https://nvidia.github.io/TensorRT-LLM/latest/legacy/performance/performance-tuning-guide/tuning-max-batch-size-and-max-num-tokens.html).
- TGI's `--max-batch-prefill-tokens` defaults to 4096
  (https://huggingface.co/docs/text-generation-inference/reference/launcher).
- SGLang sizes `chunked_prefill_size` from memory: 2048 under 20 GB, 8192 on large GPUs
  (https://docs.sglang.ai/advanced_features/hyperparameter_tuning.html). The exact tiers were not confirmed in
  source.

**Agreement:** engines treat ~2k tokens per step as enough on small GPUs and 8-16k on Hopper/Blackwell. Our
dense M\* (L4 ~1.5k, Hopper 2-7k, B200 6k to over 12k) falls in the same bands. These are the T[gpu] values in
section 4.

**Our vLLM serve sets no token budget** (`vllm_shim.serve_args`: only `--max-num-seqs 256`). Through the API
server it therefore runs at 2048 (A100), 8192 (H100, H200) or 16384 (B200) tokens per step. Two consequences:
- `jeval_concurrency` only has to keep that budget full. Eight requests of multi-thousand-token JEVal states do
  that on every GPU, which is why 8 is steady everywhere and more only adds latency.
- The budget changes with vLLM version and GPU memory, so the design records it (section 5.6).

**What utilization means** (https://docs.nvidia.com/deploy/nvml-api/structnvmlUtilization__t.html):
- nvidia-smi `utilization.gpu` is the fraction of the sample window in which at least one kernel ran.
- DCGM separates SM_ACTIVE, SM_OCCUPANCY and PIPE_TENSOR_ACTIVE; NVIDIA says SM_ACTIVE ≥ 0.8 is needed but not
  sufficient (https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/feature-overview.html).
- Our kernel_busy is the profiler version of the same "a kernel is running" signal. It shows we have left the
  host-bound region, **not** that the GPU is used well. Example: LFM2 on the L4 sits at kernel_busy 0.94 while
  doing 18-21 TFLOP/s, about 35% of the L4's measured peak.

**Consequence:** the report should keep quoting achieved TFLOP/s or MFU next to B\*, as `docs/multimodal.md`
already does. OpenJev's nvidia-smi-based steady state stays a weaker signal than kernel_busy.

### 3.3 MoE and hybrid models

**Tokens per expert ≈ M·k/E.**
- LFM2-8B-A1B has 32 experts, top-4, hidden 2048 and 24 layers (18 short-conv + 6 GQA). That makes M/8 tokens
  per expert (https://huggingface.co/LiquidAI/LFM2-8B-A1B, https://arxiv.org/html/2511.23404v1).
- Its card says 1.5B active, not 1B.
- NVIDIA says DeepSeek-V3 experts stay memory-bound up to ~10k tokens per pass
  (https://nvidia.github.io/TensorRT-LLM/latest/blogs/tech_blog/blog24_MoE_as_Dense_GEMM.html).
- For LFM2, ≥ 128-256 tokens per expert needs only M ≥ 1-2k. Our M\* is 10x that, so **host time, not expert
  GEMM shape, sets LFM2's B\***. This agrees with the formula.

**HF expert dispatch** (https://huggingface.co/docs/transformers/en/experts_interface):
- The eager backend "loops over selected experts" with a data-dependent op, which is a host sync.
- `grouped_mm` sorts tokens and uses `torch.nn.functional.grouped_mm` and is "best for larger inputs".
- Agreement: our eager MoE is ~9x slower at B = 1, issues 10.9-11.7k kernels per forward (vs 1.6-1.8k with
  grouped_mm), and is still rising at B = 64.
- New point from our records: on the **L4 and A100 the default path issues ~2.9-3.2k kernels per forward**,
  against 1.6-1.8k on sm90+. `grouped_mm` is not used there, which is why LFM2 tops out at 0.93-0.94 on those
  GPUs even at large B.

**Hybrids.**
- vLLM reports granite-4.0-h-tiny gaining up to 91% throughput only after full CUDA graphs, because Mamba and
  linear-attention kernels are launch-heavy Triton (PyTorch blog above).
- Liquid publishes no GPU throughput numbers for LFM2; its report benchmarks CPU and phones only.
- Our data agrees in direction: per layer, LFM2 issues 65-73 kernels and Qwen3-8B 55. Without graphs, hybrids
  and MoE should expect a larger M_host. The simple formula captures this only through the active-parameter
  factor. An eager per-expert loop (~6x the kernels) is outside what it models, which is one more reason the
  table decides.

### 3.4 Host overhead

- CUDA graphs remove per-launch cost: 2.7x end to end in PyTorch's Seamless M4T case
  (https://pytorch.org/blog/accelerating-generative-ai-4/). HF `grouped_mm` cannot be graph-captured, so for
  LFM2 the realistic fix is pretokenizing plus grouped_mm, which we did (0.97/0.98 on the H100).
- Tokenizers are fast per byte (~420 MB/s per core, https://arxiv.org/pdf/2511.20849). Inline tokenization caps
  busy because it runs *serially* with the GPU, not because it is slow. Agreement: inline prep is 8-12% of
  LFM2's wall time on big GPUs and under 2% on the L4. The GPU work per row is 5-10x shorter on big GPUs, so the
  same host cost weighs more there.
- **Rule of thumb:** host prep caps kernel_busy at G / (G + P), where P is serial prep per batch. **Once P > 5%
  of G, no batch size reaches 0.95**, because both grow linearly in B.
  - For text this is fixed by pretokenizing or prefetching.
  - For images, CPU preprocessing has been reported as *the* bottleneck in vLLM VLM serving on an L4
    (https://discuss.vllm.ai/t/cpu-utilization-is-extremely-high-during-inference-and-becomes-the-primary-performance-bottleneck/1876).
    vLLM moves normalization to the GPU and caches processor outputs
    (https://docs.vllm.ai/en/v0.28.0/design/mm_processing/).
  - This matches d1's 1024 px arm: 0.97 on the L4, where GPU work per image is large, and 0.87-0.91 on the H100.
    The multimodal doc suspects the unpinned `pixel_values` copy and the feature scatter; both are host-side.
  - Expected for such arms: "not reached", reported as host-bound rather than forced to a B\*.

### 3.5 Memory cap

Prefill-only inference frees each layer's activations, so the peak is the largest single transient, not
hidden x layers. The candidates are:
- full-sequence logits (B·L·V·2-4 bytes, avoided by `logits_to_keep=1`, transformers PR #31292);
- gated-MLP intermediates (B·L·d_ff·2 bytes x 2-3);
- MoE dispatch buffers;
- eager attention scores (B·h·L²·2 bytes, avoided by SDPA).

**Disagreement:** the subagent attributed the L4 OOM to logits. SemIf passes `logits_to_keep=1`, so it is not
logits.

What the records show: on the L4 (~6-7 GB free after 15-17 GB of weights):
- 8-9B dense runs OOM at ~25-42k padded tokens and survive ~21k;
- LFM2 survives 16 x 1315 (~21k) and OOMs at 32.

So the transient is **≤ ~0.25-0.3 MB per token** for these models. d1's `vl_batch` measures ~0.08 MB/token
(text) directly.

The `batch` benchmark has no peak-memory metric. Its `_mb` key is *modelled traffic*, not allocation. vLLM's
approach is to run a profiling forward at the largest shape and measure peak memory
(https://discuss.vllm.ai/t/question-about-profile-run/1194), and it does not auto-cap a too-large budget. The
design therefore:
- adds `batch_s<L>_b<B>_peak_mb` from `torch.cuda.max_memory_allocated`;
- uses a conservative 0.3 MB/token until a model has a measured value.

## 4. Recommended defaults

### 4.1 Per-GPU constants

`T[gpu]` is the tokens per forward that keep the GPU busy for an ~8B dense model, from our dense sweeps and the
engine budgets in section 3.2. Rows marked *proposed* have no records yet.

| GPU (canonical key) | T (tokens/forward) | Usable memory | Basis |
|---|---|---|---|
| L4 | 1,500 | 21.5 GB | measured (dense B\* 1-4) |
| A10G | 1,500 | 21.5 GB | proposed: L4-class FLOP/s; vLLM/SGLang 2k tier |
| RTX Pro 4500 | 2,000 | 29 GB | proposed: between L4 and A100 until a ceiling run exists |
| A100-SXM4 / A100 80GB PCIe | 3,000 | 72 GB | measured (Kev-4B 16 x 320 = 5.3k; qwen3-8b flat from 0.4k) |
| L40S | 3,000 | 43 GB | proposed: measured bf16 peak 186 TFLOP/s, A100-class |
| H100, H200 | 6,000 | 72 / 127 GB | measured (dense 2.3-7k, vLLM 8-16k tier) |
| B200 | 12,000 | 162 GB | measured (qwen3-8b 128-token plateau at 64 x 660 padded; 1024 B\* 4) |

The model factor is `max(1, 8B / active_params)`: 1 for 8-9B dense models, ~2.3 for 3-4B dense, ~5.3 for
LFM2-8B-A1B (1.5B active). It never shrinks below 1, because bigger models already saturate at batch 1-4.

### 4.2 Resulting defaults

Row tokens are 128 → ~400 and 1024 → ~1,300. Defaults for model classes we have not swept on that GPU,
**B at 128 / B at 1024**, before the memory cap:

| GPU | Dense 8-9B (x1) | Dense 3-4B (x2.3) | MoE ~1.5B active (x5.3) | vLLM OpenJev-class |
|---|---|---|---|---|
| L4 | 4 / 1 | 8 / 2 | 16 / 8 | concurrency 8 |
| A10G | 4 / 1 | 8 / 2 | 16 / 8 | 8 |
| RTX Pro 4500 | 4 / 2 | 16 / 4 | 32 / 8 | 8 |
| A100 | 8 / 2 | 16 / 4 | 32 / 16 | 8 (measured) |
| L40S | 8 / 2 | 16 / 4 | 32 / 16 | 8 |
| H100 / H200 | 16 / 4 | 32 / 8 | 64 / 32 | 8 (measured) |
| B200 | 32 / 8 | 64 / 16 | 128 / 64 | 8 (measured) |

- **For vLLM:** `c = max(8, 2 * ceil(max_num_batched_tokens / mean_prompt_tokens))`, capped at
  `max_num_seqs`. This assumes prompts of a few thousand tokens; short-prompt vLLM models would need larger
  values.
- **Very short rows overshoot.** For d1-3B's ~60-token text rows the formula gives 64 on the L4 and 256 on the
  H100, against measured B\* 32 and 64. Those pairs are in the table, so the formula is not used for them, but
  short-row models on new GPUs should get a `steady-bracket` run.

### 4.3 Expected kernel_busy at the default (from the back-test)

- **Dense models with HF eager or CUDA graphs:** 0.95-1.0 on L4/A100/H100/H200 and 0.93-1.0 on B200.
- **MoE with pretokenized grouped_mm on sm90+:** 0.95-0.96. With inline prep: 0.82-0.94, i.e. "not reached"
  (host prep). On sm80/89 (no grouped_mm): ≤ 0.94.
- **Kev-4B on Hopper/Blackwell:** 0.86 at 128 tokens, 0.23-0.30 at 1024. Not reachable by batching; still
  unexplained, needs a trace.
- **Laya:** ≤ 0.92 everywhere.
- **d1 image arms on Hopper:** 0.87-0.91.

### 4.4 Back-test detail

Pick = the simple formula, with active parameters estimated per series as FLOPs per token / 2 from the record,
and the T values above. Compared with the throughput at the measured B\* (or the plateau), across 36 series:

| Group | Count | Throughput at the pick vs reference |
|---|---|---|
| All series | 31 / 36 | ≥ 95% |
| B200 qwen3-8b 128 | 1 | 95% after rounding (pick 32, plateau 64) |
| A100 qwen3-8b | 2 | 91-92% (pick 8 / 2, B\* 1) |
| L4 lfm2-8b-a1b 1024 (inline) | 1 | 92% (pick 8, plateau 1) |
| jevk5-9b L4 128 | 1 | 82% (pick 4, B\* 2). Its twin qwen35-9b gives 100% at B = 4, so this is run noise. |

Exact B\* matches include the H100 pretokenized LFM2 at 1024 (16), the H100/H200 qwen3-8b at 1024 (4), and
every L4 9B dense at 1024 (1).

### 4.5 The detailed version (not used)

An earlier draft modelled host time explicitly:

```
M* = max(T[gpu], 19 * H * F_eff / F_tok)        # H = kernels per forward x ~6 µs per kernel
```

- `F_eff` is 0.6 x the GPU's measured bf16 peak (L4 34, A100 155, H100/H200 404, B200 804 TFLOP/s).
- `F_tok` is FLOPs per token from `flops.py`.
- `H` comes from the record's B = 1 `launch_api_ms` at the shortest length. At long lengths the launch API
  blocks on a full queue, which inflates it (L4 qwen 1024: 152 ms instead of 20-31 ms).

It also scored 31 of 36 with an 82% worst case, but it needs kernel counts per model and runtime and a per-kernel
host cost that varies 3x across Modal hosts (2.2-8 µs). The simple version folds all of that into the 8B
reference, because H is about the same (10-30 ms) for every model we run. Keep the detailed form in mind only for
a runtime whose host cost is very different, e.g. full CUDA graphs (smaller H) or eager per-expert MoE loops
(~6x the kernels); the bracket run will show it.

## 5. Code design

### 5.1 Critique of the starting design

The parent's design is a pinned table keyed by (model, GPU, length bucket). It resolves `batch = "steady"` in
`runner.run` after `detect_gpu()`, has a recompute test, and fails fast on unknown combinations. That is the
right skeleton. Changes needed:

1. **B differs per state length inside one run.** `batch` walks `[grid] batch_state_tokens`, so the runner
   cannot write one `batch` value. It must resolve to a per-length mapping, and the benchmark must run only
   those (L, B_L) points.
2. **The batch is a `[grid]` axis (`batch_sizes`), not a param.** The config syntax has to live there.
3. **The key needs the variant.** B\* depends on `batch_host_prep`, `semif_experts_implementation`,
   `laya_mode` and fused/graphs. LFM2 on the H100 is B\* 32 pretokenized but "not reached" inline. An entry
   records the settings it was measured under. The resolver applies them by default, and refuses (or falls back
   to the formula) when the config overrides one.
4. **"Not reached" needs a status.** Plateau entries are legitimate picks, but a record must not claim 95%.
5. **GPU key normalization.** `detect_gpu()` returns `"NVIDIA H100 80GB HBM3"`, multi-GPU strings, and labels
   that differ from the requested one (gpu-parallel L4 vs A10G). The table must distinguish A100 PCIe from SXM4
   (2.22 vs 2.49 q/s) and include GPU count for tensor-parallel vLLM.
6. **`jeval_concurrency = "steady"` is meaningless for SemIf-family backends,** whose forwards serialize under
   `SemIf.lock`. Only server-batched backends (`vllm_shim`) get it. Others raise.
7. **The formula needs the model's HF config and its loaded weight bytes.** Table lookup can happen before
   model load (so the default policy still fails fast). Formula resolution happens right after load and before
   the first benchmark.
8. **Fallback choice:** the table decides, and a missing pair fails by default (the parent design's fail-fast
   stays the default). The formula is opt-in, and online calibration is rejected (section 5.7).

### 5.2 Module: `src/s1/steady_state.py`

Torch-free, importable on the login node.

```python
RULE = 1  # bump when the table schema, the formula or its constants change (recorded as steady_rule)

Knob = Literal["batch", "concurrency"]
Status = Literal["reached", "plateau"]  # plateau: B* not reached, the throughput plateau is used
Source = Literal["table", "formula", "explicit"]


@dataclass(frozen=True)
class Entry:
    model: str  # catalog name, e.g. "lfm2-8b-a1b.zeroshot"
    gpu: str  # canonical key, see GPUS
    knob: Knob
    at: Mapping[int, int]  # state length -> B (batch); {0: c} for concurrency
    status: Mapping[int, Status]
    variant: Mapping[str, Scalar]  # settings it was measured with, e.g. {"batch_host_prep": "pretokenized"}
    runs: tuple[str, ...]  # source run ids; the test recomputes `at` from them
    gpus: int = 1  # tensor-parallel width (vLLM)
    note: str = ""


@dataclass(frozen=True)
class Gpu:
    tokens: int  # T[gpu]: tokens per forward that keep it busy for an ~8B dense model
    mem_gb: float  # usable for weights + activations
    measured: bool  # False: proposed defaults (section 4)


@dataclass(frozen=True)
class Pick:
    knob: Knob
    at: Mapping[int, int]  # length -> B (or {0: c})
    source: Source
    status: Mapping[int, Status | Literal["predicted"]]
    runs: tuple[str, ...]  # table: source runs; formula: runs its inputs came from
    # formula inputs for the record: T, active_b, row_tokens, B_mem
    inputs: Mapping[str, float]


TABLE: tuple[Entry, ...]
GPUS: Mapping[str, Gpu]
NO_BATCH: Mapping[str, str]  # moved here from report_steady (report imports it)
SERVER_BATCHED: frozenset[str] = frozenset({"vllm_shim"})
QUESTION_TOKENS = 270  # JEVal question + options, added to the reference state length
REF_ACTIVE_B = 8.0  # the model size T[gpu] is quoted for
BUSY = 0.95  # imported by report_steady (single source of truth)


# "NVIDIA H100 80GB HBM3" -> "H100"; "NVIDIA A100 80GB PCIe" -> "A100-PCIe"; KeyError if unknown
def gpu_key(detected: str) -> str: ...
def lookup(model: str, gpu: str, settings: Settings, *, knob: Knob, gpus: int = 1) -> Entry | None: ...
def formula(
    gpu: str,
    lengths: Sequence[int],
    *,
    active_params: float,
    weights_gb: float,
    mem_mb_per_token: float = 0.3,
) -> Pick: ...
def mem_cap(gpu: str, weights_gb: float, row_tokens: float, mb_per_token: float) -> int: ...
def bracket(b: int) -> list[int]: ...  # [b//2 or 1, b, 2b], deduplicated
def resolve_pre(spec: RunSpec, gpu: str) -> tuple[RunSpec, Pick | None]: ...  # table only, before model load
# formula, after load
def resolve_post(spec: RunSpec, gpu: str, model: object) -> tuple[RunSpec, Pick | None]: ...
def record_fields(pick: Pick) -> dict[str, Scalar]: ...
```

**Length buckets.** The table holds lengths 128 and 1024 (and d1's ~60/1.8k rows). A run length L resolves as
follows:
- to the entry's value when L is within a factor of 2 of a measured length (so 64-256 use the 128 entry and
  512-2048 the 1024 entry);
- otherwise it counts as a missing entry: an error under `"table"`, the formula under `"table+formula"`.

The table stays the answer for every length near what we measured, without pretending we measured lengths far
from it.

**Formula inputs:**
- `active_params`: `flops.arch(cfg).active_params` from the loaded model's HF config (for an MoE, the
  parameters one token uses, not the total).
- `T[gpu]`: `GPUS[gpu].tokens`.
- `weights_gb`: `torch.cuda.memory_allocated()` after load, passed from the runner through a model hook
  (`getattr(model, "weights_gb", None)`). Otherwise the catalog's `weights_gb`.

### 5.3 Config syntax

```toml
# batch / vl_batch: per-length steady batch
[grid]
batch_state_tokens = [128, 1024]
batch_sizes = ["steady"]            # or ["steady-bracket"]: {B/2, B, 2B} at each length, to verify the pick

[params]
steady_policy = "table"             # default: measured entries only, fail fast otherwise
# steady_policy = "table+formula"   # opt-in: table first, formula (status "predicted") only for missing pairs

# OpenJev (vllm_shim only)
[params]
jeval_concurrency = "steady"
```

Explicit integers keep working unchanged: an explicit sweep *is* how table entries get made. Mixing
`"steady"` with integers in one axis is an error.

### 5.4 Where it hooks in

1. **`config.load`**
   - Accept the string sentinels above in `batch_sizes`, `vl_batch_sizes` (grid) and `jeval_concurrency`
     (params/sweep).
   - Validate `steady_policy`.
   - Keep `RunSpec` otherwise as is.
2. **`runner.run`, right after `invariants["gpu"] = detect_gpu()` (`src/s1/runner.py:~129`)**
   - Call `steady_state.resolve_pre(spec, invariants["gpu"])`.
   - With no entry under the default `steady_policy = "table"`, raise `SteadyStateError` here, before the model
     loads. The message names the nearest entries and the sweep config to run.
   - With a hit, rewrite the spec:
     - `grid["batch_sizes"]` becomes the sorted union of the per-length values, so `report_batch` keeps working.
     - Add `params["batch_sizes_at"] = "128:32,1024:16"`.
     - Apply the entry's `variant` settings via `setdefault`, so a run reproduces the conditions the entry was
       measured under. A conflicting explicit setting (say, inline prep against a pretokenized entry) means
       the entry does not apply: an error under `"table"`, the formula under `"table+formula"`.
3. **`runner.run`, right after the model is built and spec params are merged (`~line 160`)**
   - Call `steady_state.resolve_post(...)` for anything still `"steady"`.
   - Recompute `settings` after this point. Today it is computed before model build, so move the `settings =`
     line or rebuild it.
4. **`benchmarks/batch_sweep.run`**
   - If `batch_sizes_at` is present, iterate only `(L, B_L)`, or the bracket set for `steady-bracket`.
   - Add `batch_s<L>_b<B>_peak_mb` at every point: `torch.cuda.reset_peak_memory_stats()` before the timed calls
     and `max_memory_allocated()` after. Only when torch is loaded, so fakes stay torch-free.
   - On OOM at a steady point, halve once and record `batch_s<L>_steady_fallback = <B/2>` before giving up.
   - Headline unchanged.
5. **`benchmarks/vl_batch.run`** (on the multimodal branch; after it merges): the same with `vl_batch_sizes_at`
   keyed by arm and px.
6. **`benchmarks/jeval`**: no change. The runner has already replaced `"steady"` with an int.
   **`vllm_shim.build`**: parse vLLM's startup log for the effective `max_num_batched_tokens` and return it from
   `run_info()` as `env_vllm_max_num_batched_tokens`.
7. **`report_steady`**
   - Import `BUSY` and `NO_BATCH` from `steady_state`.
   - Add a "Picked by" column (table or formula, plus the source runs).
   - Add a drift check: for `steady-bracket` records, recompute B\* from the three points and flag it when B\*
     ≠ the pick, or when 2B beats B by ≥ 5%.

### 5.5 Record fields

All go in `params`, because they affect the numbers (AGENTS rule 3):
- `steady_rule` (int, `RULE`).
- `steady_source` (`table` | `formula`).
- `steady_runs` (comma-joined run IDs).
- `steady_status` (`128:reached,1024:plateau` or `predicted`).
- `batch_sizes_at` / `vl_batch_sizes_at`, or the resolved int `jeval_concurrency`.
- `steady_requested` (the original sentinel, so a reader can tell an automatic pick from a hand-set one).
- For formula picks, `steady_inputs` as compact `key=value` text: `T`, `active_b`,
  `row_tokens`, `B_mem`.
- Plus `batch_s<L>_b<B>_peak_mb` in results and `env_vllm_max_num_batched_tokens` in invariants.

### 5.6 Tests (all on the login node, no GPU)

1. **`test_table_matches_records`.** For every `Entry`, load its `runs` from `results/` and recompute
   `report_steady.steady(...)` (or `ConcCurve.steady` for concurrency). Assert `b_star` (status reached) or
   `plateau` (status plateau) equals `at[L]`. The entry cannot drift from its evidence; a newer sweep must be
   added explicitly.
2. **`test_entries_are_buildable`.** Every entry's model is in the catalog. `knob == "batch"` implies the
   backend class defines `decide_batch`. `knob == "concurrency"` implies the backend is in `SERVER_BATCHED`.
   Every `NO_BATCH` model lacks `decide_batch`.
3. **`test_gpu_key`.** Every `invariants.gpu` string in `results/` maps to a key or is explicitly ignored (cpu).
   The PCIe and SXM4 A100s map apart.
4. **`test_formula_backtest`.** For every measured `batch` series, the formula pick reaches ≥ 0.80 of the
   reference throughput (today's minimum is 0.82). This is a regression guard on the constants.
5. **`test_formula_properties`.** B ≥ 1, B is non-increasing in L, B ≤ `mem_cap`, B is a power of two, and
   a larger `T[gpu]` or a smaller `active_params` never lowers B.
6. **Runner tests** (`EchoModel` plus a fake `detect_gpu`):
   - `"steady"` resolves and writes the record fields.
   - The default policy with an unknown pair raises *before* the model factory is called (assert with a
     factory that fails if called).
   - A table entry always wins over the formula, even under `"table+formula"`.
   - A variant mismatch raises under `"table"` and falls back to the formula under `"table+formula"`.
   - `jeval_concurrency = "steady"` on a non-server backend raises.
7. **Config tests.** The sentinels parse, mixed lists are rejected, and any Modal config using `"steady"` with
   the default policy has an entry for every `[launch] gpu`.
8. **`batch_sweep` tests.** `batch_sizes_at` restricts the points, the bracket set is right, `peak_mb` is absent
   under fakes, and halving on OOM is recorded.

### 5.7 Why not online calibration

A mini-sweep at run start (e.g. B ∈ {1, 4, 16, 64} for a few batches each, then pick) was considered and is
**not recommended as a default**:

- **Noise.** kernel_busy jitters by ±0.03-0.05 between neighbouring B values in one run (qwen3-8b on Hopper dips
  to 0.90-0.91 at B = 16 between 0.95 at 8 and 32). A per-run pick would flip between runs of the same pair,
  and the comparison is only fair if the batch is the same every time. That is what pinning (AGENTS rule 3)
  protects.
- **Cost.** A short sweep is 25-50% of a full `batch` run (the warm-up per shape dominates), paid on every run.
- **Confounds.** The pick would bake in that container's host CPU (2.2-8 µs per kernel across Modal hosts, a 3x
  spread) and the sm90 cuDNN plan-per-shape cost at short lengths. Both should be visible in the record, not
  absorbed silently into B.
- **Better alternative.** `steady-bracket` spends three points instead of 7-9, records them, and lets the report
  and test flag drift. A verified pair then becomes a table entry by a reviewed edit.

## 6. Failure modes

| Failure | What happens | Mitigation in the design |
|---|---|---|
| GPU label ≠ GPU (gpu-parallel "L4" is A10G; `gpu` got an A10G) | Wrong entry | Resolve from `detect_gpu()`, never the launch label; `require_gpu` still fails fast |
| Host CPU differs from the entry's run (Zen 5 vs Skylake) | B\* may shift one step | Record `cpu_model`; bracket mode |
| Variant differs (inline vs pretokenized, eager experts, graphs off) | Entry invalid (LFM2: B\* 32 vs not reached) | Variant is part of the key; the entry's settings are applied by default; a conflicting override → error (or formula when opted in) |
| Length not near a measured bucket | Extrapolation | Formula scaled by row tokens; status `predicted` |
| Real rows longer than the reference (padding: 37% at B = 32, L = 128) | More tokens than planned, possible OOM | Memory cap uses padded tokens; `peak_mb` recorded; halve-once fallback |
| B\* "not reached" (LFM2 inline, Laya, Kev-4B Hopper 1024, d1 images on H100) | A batch is picked but busy < 0.95 | Status `plateau` in the record; the report must not claim steady state |
| Single-run noise in the source sweep (jevk5 vs qwen35 differ at 128) | Entry one step off | Test only checks consistency with records; drift check via bracket runs; entries cite every run |
| vLLM upgrade changes the default token budget | Steady concurrency may move | Record `env_vllm_max_num_batched_tokens`; rule `c ≥ 2 x budget / mean prompt` |
| SemIf `jeval_concurrency` (lock-serialized) | "steady" would be meaningless | Rejected for non-server backends |
| sm90+ cuDNN SDPA plan building per new shape | Host-confounded short-length B\* | Noted on Hopper/Blackwell entries; an sdpa_backend A/B is still owed (memory: modal-host-cpu-confound) |
| Formula constants (`T[gpu]`, the 8B reference) wrong for a new GPU (only under `"table+formula"`) | Under- or overshoot | `measured=False` GPUs are flagged in the record; backtest test; one sweep turns them into measured |
| Table entry superseded by a newer sweep | Old pick persists | Intentional: changing it is a reviewed edit with new `runs` (and a `RULE` bump only if the schema changes) |

## 7. Sweeps still needed (not run; estimates only)

Rates are approximate Modal list prices including CPU and memory, scaled from the estimates in the existing
configs: L4 ≈ $1/h, A100-80GB ≈ $3/h, H100 ≈ $4.5/h, H200 ≈ $5/h, B200 ≈ $7/h, L40S ≈ $2.2/h. Check them
against the guard ledger before launching.

| # | Sweep | Why | GPUs | Est. time | Est. cost |
|---|---|---|---|---|---|
| S1 | lfm2-8b-a1b **pretokenized** (grouped_mm), B 1-256, L 128/1024 | Only the H100 has a reached entry; the others have inline plateaus only | A100-80GB, H200, B200 | ~20 min each | ~$4 |
| S2 | qwen35-9b and jevk5-9b, B 1-64 | The L4's largest models have no big-GPU entries; qwen3-8b is only a proxy | A100-80GB, H100, H200, B200 | ~12 min per model per GPU | ~$5 |
| S3 | d1-3B text arm (`vl_batch` arm `none` and `text`), B 1-256 | Fills d1 entries beyond L4/H100; checks the formula's 3B dense class | A100-80GB, H200, B200 | ~15 min each | ~$3 |
| S4 | OpenJev on L40S (fp8, TP1), `jeval_concurrency` [1, 4, 8, 16, 32], 500 questions | Largest model that fits a 48 GB card; first sm89 vLLM point | L40S | ~60-75 min | ~$2.5 |
| S5 | qwen35-9b / jevk5-9b / lfm2 on A10G and RTX Pro 4500 | GPUs with only proposed constants | HUIT Slurm (`gpu-spec --gpu-type a10g`, `gpu-g7-reserved`) | ~45 min each | $0 Modal |
| S6 | One `steady-bracket` run per new table entry (after implementation) | Exercises the drift check | as needed | ~5 min | ~$0.5-1 |

- **Total Modal:** about **$15** (S1-S4 plus S6), and every run also starts recording `peak_mb`.
- **Not worth sweeping for B\*:** Laya and the Kev-4B 1024-token floor on Hopper/Blackwell. They need a trace,
  not more batch sizes.
- **Every new sweep uses `gpu_metrics = "detailed"`,** as the Modal-config test already enforces.

## 8. Follow-ups found along the way (outside this design)

- `batch` lacks a peak-memory metric (`_mb` is modelled traffic). Included above because the memory cap needs
  it.
- LFM2 on sm80/89 issues ~2x the kernels of sm90+. Worth confirming which expert path transformers takes there
  (`env_experts_implementation`) before attributing the A100's 0.93 ceiling to the GPU.
- The brief for this design listed Kev-4B and Laya as "batch-capable but never swept". They were swept on
  2026-11-09 (`results/steady_state_findings.md` already uses those records): Laya never reaches 0.95, and
  Kev-4B reaches it only on the L4 and A100.
