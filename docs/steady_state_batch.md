# Steady-state batch size: what to use, and how runs pick it

We compare models only at **steady state**: the GPU busy at least 95% of the time, not stalled waiting for the
CPU. How large a batch that takes depends on the model, the GPU and the prompt length. This page gives the
measured answer for each pair and explains how future runs pick it automatically. The code is
`src/s1/steady_state.py`; the design and the research behind it are in `docs/steady_state_batch_design.md`.

## How we decide a steady-state batch (a sweep)

1. Run the model at batch 1, 2, 4, 8, ... (the `batch` benchmark, or `jeval_concurrency` for vLLM).
2. At each batch, measure how busy the GPU is (`kernel_busy`: GPU kernel time / wall time, from a
   torch.profiler trace) and how fast it goes (decisions per second).
3. The steady-state batch **B\*** is the **smallest batch where the GPU is >= 95% busy and no bigger batch is
   >= 5% faster** (`report_steady.steady`).
   - Busy alone is not enough: the GPU can be busy with small, inefficient work while a bigger batch still
     gets more done.
   - Speed alone is not enough: throughput can flatten while the CPU still leaves the GPU idle.
4. When no batch reaches 95% (the CPU holds the GPU back however large the batch), the table records the
   **plateau** instead: the smallest batch no bigger one beats by 5%. Runs at a plateau batch are *not* at
   steady state, and the record says so.

For OpenJev on vLLM there is no kernel trace, so the busy signal is nvidia-smi's per-GPU utilization, and
the knob is the number of concurrent requests.

## What batch size to use

Default setup, **batch at 128-token / 1024-token states**. `*` = 95% busy not reached (plateau batch);
`-` = not measured yet.

| Model | L4 | A100-SXM4 | A100-PCIe | H100 | H200 | B200 |
|---|---|---|---|---|---|---|
| qwen3-8b.zeroshot | 4 / 1 | 1 / 1 | - | 4 / 4 | 4 / 4 | 64 / 4* |
| qwen35-9b.zeroshot | 4 / 1 | - | - | - | - | - |
| jevk5-9b | 2 / 1 | - | - | - | - | - |
| kev-4b.hybrid | 8 / 1 | 16 / 2* | - | 16 / 2* | 16 / 2* | 16 / 8* |
| lfm2-8b-a1b.zeroshot | 2 / 1* | 32 / 32* | 32 / 32 (overlapped) | 32 / 16 (overlapped) | 32 / 8 (overlapped) | 32 / 16 (overlapped) |
| kev-4b.qwen3 | 1 / 1* | - | - | - | - | - |
| olmoe-1b-7b.zeroshot | 2 / 1* | - | - | - | - | - |
| laya.multilingual | 8 / 4* | - | - | - | - | - |
| laya.english | 8 / 4* | 32 / 32* | - | 64 / 32* | 64 / 32* | 32 / 16* |
| openjev | - | - | concurrency 8 | concurrency 8 | concurrency 8 | concurrency 8 |
| d1-3b.eager | text 32 / 1024 px image 1 | - | - | text 64 / 1024 px image 32* | - | - |

- Rows are ~270 tokens longer than the state (the JEVal question and options), so "128 tokens" is about 400
  tokens per request and "1024" about 1,300.
- A big model on a slow GPU is busy from batch 1-4 (9B models on the L4). A small model on a fast GPU needs
  many requests at once (LFM2, 1.5B active parameters, needs 32 on the H100).
- LFM2 only reaches steady state when prompt prep stays off the GPU's critical path. With
  `batch_host_prep = "overlapped"` (batch i+1 is tokenized on a CPU thread while batch i runs; the prep stays
  inside the wall time, as in a serving loop) it reaches 95% busy on the A100 PCIe, H100, H200 and B200, so
  that is the setup the table picks there (records from #21). `"pretokenized"` gives the same batch on the
  H100. With inline prep it only reaches a plateau.
- granite4-h-tiny has no row: on the current L4 image (no mamba-ssm kernels) every batch >= 2 runs out of
  memory.
- **d1-3B** (vision-language) is measured on the `vl_batch` benchmark, not `batch`, so its cell reads
  "text rows alone / with a 1024 px picture". A picture adds ~1.8k tokens: on the L4 one such request already
  fills the GPU (batch 1), while ~60-token text rows need 32. On the H100 the image arm never reaches 95% busy
  (host-side image prep), so 32 is its fastest batch. Per image size, see "d1-3B" below.
- Not here: elephant-1 and bongard-mini (no batched path); `"steady"` is an error for them.

## How future runs pick the batch

Say `"steady"` instead of a number:

```toml
[grid]
batch_state_tokens = [128, 1024]
batch_sizes = ["steady"]          # or ["steady-bracket"] to also run half and double, as a check

[params]
jeval_concurrency = "steady"      # vLLM-served models (OpenJev) only
```

For d1-3B on `vl_batch`, the same word goes on its size axis; the pick is per arm (image, text, no picture)
and image size, and must match a measured size exactly (384, 768 or 1024 px):

```toml
[grid]
vl_batch_image_px = [384, 1024]
vl_batch_sizes = ["steady"]
```

When the run starts, the runner:

1. **Checks which GPU it really got** (`nvidia-smi`, not the requested label; Slurm labels are sometimes
   wrong).
2. **Looks up the table** for that model, setup and GPU, and takes the value for the nearest measured
   length within 2x (64-256 tokens use the 128 entry, 512-2048 the 1024 entry). If the entry was measured
   with a particular setup (overlapped prompt prep, a Laya mode), the run uses that setup unless the config sets
   it.
3. **Runs only that batch** at each length (or half, it and double with `"steady-bracket"`).
4. **Records what it chose and why** in the run's params: `batch_sizes_at` (e.g. `"128:32,1024:16"`;
   `vl_batch_sizes_at` for d1-3B, e.g. `"image@1024:1,none@0:32"`),
   `steady_source` (`table`), `steady_status` (`reached` or `plateau` per length), `steady_runs` (the sweep
   it came from), `steady_gpu` and `steady_rule`.

**If the pick runs out of memory**, the benchmark halves it (B/2, B/4, ... down to 1) until a batch fits,
measures that, and records `batch_s<L>_steady_fallback` = the batch it used (0 if none fit) next to the
pick's `_oom` = 1. Why the dynamic design needs this:

- **The pick is a lookup, not a guarantee for this container.** Entries come from one past sweep. A new run
  can carry more memory than that one did: longer real rows or more padding than the reference states, a
  different setup applied by the config, or a smaller card behind the same key (`A100-PCIe` covers 40 GB and
  80 GB cards). The opt-in formula's memory cap is itself an estimate (0.3 MB per token).
- **A "steady" run measures one batch per length.** A sweep that hits OOM at B = 64 still has B = 1-32. A
  steady run that hits OOM at its only point would record nothing for that length and waste the GPU time,
  including the model load.
- **Halving keeps the run useful and honest.** Steady state is reached at the smallest busy batch, and
  throughput is flat just above it, so the largest batch that fits is the best available stand-in. The record
  says it is no longer the table's pick, so it is not quoted as one. A fallback is a signal to re-sweep that
  pair and update the table.

Explicit sweeps and `"steady-bracket"` runs do not halve: their other sizes already cover the smaller
batches, and the OOM point is itself data.

**If the table has no entry** for that model x GPU x setup, the run stops *before the model loads*, with a
message saying what is missing. Fill the gap with a sweep (below), or set the batch explicitly.

**Opt-in estimate.** `steady_policy = "table+formula"` lets a missing pair run on an estimate instead:

```
tokens = T[gpu] x max(1, 8B / active parameters)       T: L4 1,500, A100 3,000, H100/H200 6,000, B200 12,000
batch  = tokens / tokens per request, rounded to a power of 2, capped by memory
```

In words: every forward pass costs the CPU a fixed 10-30 ms to launch, so the GPU needs much more work than
that per pass. Start from the GPU's number, scale it up for models smaller than ~8B active parameters (less
GPU work per token), and divide by the request length. The table always wins where it has an entry; formula
picks are recorded as `predicted` and are for exploring new GPUs, not for results to cite. On the 36 measured
series this estimate is within 5% of the best throughput in 31 (worst 82%).

**Checking a pick.** `batch_sizes = ["steady-bracket"]` runs half, the pick and double. If double is >= 5%
faster or the pick is no longer 95% busy, re-sweep and update the table.

## Adding or updating an entry

1. Sweep the pair with explicit sizes (e.g. `batch_sizes = [1, 2, 4, 8, 16, 32, 64]`, as in
   `configs/steady_state_batch_l4.toml`) and commit the record.
2. Add a row to `_BATCH` (or `_CONCURRENCY`) in `src/s1/steady_state.py` with the run ID and setup.
3. Run `uv run python -c "from s1 import steady_state; print(steady_state.summary_table());
   print(); print(steady_state.markdown_table())"` and paste the output into this page.
   `tests/test_steady_state.py` recomputes every row from its record and checks this page matches the code.

## Full table

Every entry, with the setup it was measured under and its source run (all on Modal).

| Model | GPU | Setup | Batch at 128 tokens | Batch at 1024 tokens | Steady state reached? | Source run |
|---|---|---|---|---|---|---|
| qwen3-8b.zeroshot | L4 | inline prep | 4 | 1 | yes | 20261109-231713-a38ec98c |
| qwen35-9b.zeroshot | L4 | inline prep | 4 | 1 | yes | 20261119-225425-e0d0badf |
| jevk5-9b | L4 | inline prep | 2 | 1 | yes | 20261119-230530-ed7206a6 |
| kev-4b.hybrid | L4 | default | 8 | 1 | yes | 20261109-230438-4ddb0b2d |
| lfm2-8b-a1b.zeroshot | L4 | inline prep | 2 | 1 | no (plateau) | 20261119-224843-9162d873 |
| kev-4b.qwen3 | L4 | default | 1 | 1 | 128: plateau, 1024: reached | 20261122-174224-2b9402d0 |
| olmoe-1b-7b.zeroshot | L4 | inline prep | 2 | 1 | 128: plateau, 1024: reached | 20261122-182750-d8dc8419 |
| laya.multilingual | L4 | eager mode | 8 | 4 | no (plateau) | 20261122-182531-9bfa76ff |
| laya.english | L4 | eager mode | 8 | 4 | no (plateau) | 20261109-205307-30a4666a |
| laya.english | L4 | graphs mode | 4 | 4 | no (plateau) | 20261109-205548-b5716441 |
| laya.english | L4 | compile mode | 8 | 16 | no (plateau) | 20261109-205821-652d33c6 |
| laya.english | L4 | compile_graphs mode | 16 | 16 | no (plateau) | 20261109-210356-5f94a8be |
| qwen3-8b.zeroshot | A100-SXM4 | inline prep | 1 | 1 | yes | 20261109-231247-cc3371f7 |
| kev-4b.hybrid | A100-SXM4 | default | 16 | 2 | 128: reached, 1024: plateau | 20261109-230438-7a7d5a81 |
| lfm2-8b-a1b.zeroshot | A100-SXM4 | inline prep | 32 | 32 | no (plateau) | 20261119-222604-2a4972b9 |
| lfm2-8b-a1b.zeroshot | A100-PCIe | overlapped prep | 32 | 32 | yes | 20261122-181032-8ea3ce5e |
| laya.english | A100-SXM4 | eager mode | 32 | 32 | no (plateau) | 20261109-230437-6351524c |
| laya.english | A100-SXM4 | graphs mode | 32 | 32 | no (plateau) | 20261109-230556-bc2c7922 |
| laya.english | A100-SXM4 | compile mode | 32 | 32 | no (plateau) | 20261109-230701-057912aa |
| laya.english | A100-SXM4 | compile_graphs mode | 32 | 32 | no (plateau) | 20261109-230957-e4075dcf |
| qwen3-8b.zeroshot | H100 | inline prep | 4 | 4 | yes | 20261109-211259-b63a8b3e |
| lfm2-8b-a1b.zeroshot | H100 | overlapped prep | 32 | 16 | yes | 20261122-175440-dc068906 |
| lfm2-8b-a1b.zeroshot | H100 | pretokenized prep | 32 | 16 | yes | 20261119-230040-acb6fcff |
| lfm2-8b-a1b.zeroshot | H100 | inline prep | 32 | 16 | no (plateau) | 20261119-222604-5a3f7682 |
| kev-4b.hybrid | H100 | default | 16 | 2 | no (plateau) | 20261109-205016-9d8b32e6 |
| laya.english | H100 | eager mode | 64 | 32 | no (plateau) | 20261109-230439-caedab1a |
| laya.english | H100 | graphs mode | 64 | 16 | no (plateau) | 20261109-230623-02c3b499 |
| laya.english | H100 | compile mode | 64 | 32 | no (plateau) | 20261109-230730-559ab4d1 |
| laya.english | H100 | compile_graphs mode | 64 | 32 | no (plateau) | 20261109-231109-e21280ed |
| qwen3-8b.zeroshot | H200 | inline prep | 4 | 4 | yes | 20261109-231248-7eb18713 |
| lfm2-8b-a1b.zeroshot | H200 | overlapped prep | 32 | 8 | yes | 20261122-181037-5b2031f6 |
| lfm2-8b-a1b.zeroshot | H200 | inline prep | 32 | 16 | no (plateau) | 20261119-222601-ced5a93f |
| kev-4b.hybrid | H200 | default | 16 | 2 | no (plateau) | 20261109-230438-b785789f |
| laya.english | H200 | eager mode | 64 | 32 | no (plateau) | 20261109-230438-3115efc1 |
| laya.english | H200 | graphs mode | 64 | 64 | no (plateau) | 20261109-230603-94e3dd99 |
| laya.english | H200 | compile mode | 32 | 32 | no (plateau) | 20261109-230702-5f3feeec |
| laya.english | H200 | compile_graphs mode | 32 | 32 | no (plateau) | 20261109-230958-c55ccaa2 |
| qwen3-8b.zeroshot | B200 | inline prep | 64 | 4 | 128: plateau, 1024: reached | 20261109-231135-fd87886c |
| lfm2-8b-a1b.zeroshot | B200 | overlapped prep | 32 | 16 | yes | 20261122-181855-63c8017d |
| lfm2-8b-a1b.zeroshot | B200 | inline prep | 32 | 16 | no (plateau) | 20261119-222600-33e72d92 |
| kev-4b.hybrid | B200 | default | 16 | 8 | no (plateau) | 20261109-230440-c7b126b6 |
| laya.english | B200 | eager mode | 32 | 16 | no (plateau) | 20261109-230438-6835673b |
| laya.english | B200 | graphs mode | 32 | 16 | no (plateau) | 20261109-230607-9ea70f2f |
| laya.english | B200 | compile mode | 64 | 16 | no (plateau) | 20261109-230700-7702ff03 |
| laya.english | B200 | compile_graphs mode | 32 | 32 | no (plateau) | 20261109-230930-a0cb6370 |

| Model | GPU | Concurrency | Steady state reached? | Source runs |
|---|---|---|---|---|
| openjev | A100-PCIe | 8 | yes | 20261119-222616-4119a7fc .. 20261119-232521-b90e1912 |
| openjev | H100 | 8 | yes | 20261119-222601-d16bbfd0 .. 20261119-225651-791ecf60 |
| openjev | H200 | 8 | yes | 20261119-222601-cb08832a .. 20261119-225427-4bc2fe3b |
| openjev | B200 | 8 | yes | 20261119-222602-58b3adbb .. 20261119-224749-e9aa1791 |

## d1-3B (`vl_batch`)

Batch per arm and image size. "image" is the row with its picture, "text" the picture replaced by
text of the same token count, "none" the row alone (~60 tokens). From `docs/multimodal.md`.

| Model | GPU | Arm | No picture | 384 px | 768 px | 1024 px | Steady state reached? | Source run |
|---|---|---|---|---|---|---|---|---|
| d1-3b.eager | L4 | image |  | 16 | 8 | 1 | yes | 20261120-141226-e1dd7b35 |
| d1-3b.eager | L4 | text |  | 16 | 8 | 1 | yes | 20261120-141226-e1dd7b35 |
| d1-3b.eager | L4 | none | 32 |  |  |  | yes | 20261120-141226-e1dd7b35 |
| d1-3b.eager | H100 | image |  | 64 | 64 | 32 | no (fastest batch) | 20261120-141037-d11b3b27 |
| d1-3b.eager | H100 | text |  | 64 | 32 | 4 | yes | 20261120-141037-d11b3b27 |
| d1-3b.eager | H100 | none | 64 |  |  |  | yes | 20261120-141037-d11b3b27 |
