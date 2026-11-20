# Multimodal decision models: what an image in the state costs

Question (project call, 2026-11-19): at the same parameter count and on the same GPU, does a multimodal decision
model carry a bigger state, put more pressure on memory, and decide less accurately (same capacity trained on
wider data)? And does it move between compute-bound and memory-bound at full GPU utilization?

Approved by the user on 2026-11-20 (experiments A-C, about $5, with an $8 cap raise).

## The models

Pins resolved read-only from Hub metadata on 2026-11-20 (`HfApi().model_info(..., files_metadata=True)`).

| Catalog name | Repo @ revision | Weights | What it is |
|---|---|---|---|
| `d1-3b.eager` | LiquidAI/d1-3B @ 05b80fa | 6.247 GB | Liquid's decision model: full post-train of LFM2.5-VL-3B |
| `lfm25-vl-3b.zeroshot` | LiquidAI/LFM2.5-VL-3B @ 35a118d | 6.247 GB | LFM2.5-2.6B-Base + SigLIP2-NaFlex 400M, post-trained on text and images |
| `lfm25-2.6b.zeroshot` | LiquidAI/LFM2.5-2.6B @ 7141b56 | 5.394 GB | the same base, post-trained on text only |

The two LFM2.5 text towers have the same architecture (30 layers: 22 short-conv + 8 GQA with 8 KV heads,
hidden 2048, the same FFN width and vocabulary); their configs differ only in rope theta and context length. So
the pair is matched exactly on text capacity. The VL checkpoint adds 426,285,296 parameters (SigLIP2 tower
412.6M + projector 13.6M; `flops.vision_stacks` reproduces the Hub total to the parameter), +16% weight bytes.

d1 answers in one prefill pass per decision and caches nothing, so there is no KV cache to fill. "KV pressure"
here means the attention working set over the state in that one pass. Only 8 of the 30 LM layers attend; were
it cached, the KV would be 16 KiB per token. The other cost a picture adds is the vision tower itself:
SigLIP2 over up to 1,024 patches per 512 px tile, with bidirectional attention within each tile.

How many LM tokens a picture becomes (LFM2.5-VL processor): 512 px tiles, 64-256 tokens per tile, up to 10
tiles plus a thumbnail; d1 caps a picture at 1 MP. ImajevBench's pictures are 1536x1024, so about 1.5k tokens
each, against ImajevBench text states of about 100 tokens.

## Experiments

**A. Same capacity, wider data (accuracy, text only).** `configs/mm_pair_jeval_l4.toml`: the VL base and its
text twin zero-shot (SemIf's letter softmax) on the d1 JEVal smoke's 200 questions and on ImajevBench's text
track. One L4.

**B. Images in a labeled decision task.** `configs/imajev_d1_l4.toml`: d1-3B on ImajevBench v2.0-lite's 254
labeled rows (`benchmarks/imajev.py`), with the pictures and with them removed. Accuracy, ECE and Brier per
track, prompt tokens per decision, latency, and peak memory above the weights. Rows whose reference is Unknown
(24) are not scored: d1 has no abstain output, so this is accuracy on the answerable rows, not the benchmark's
leaderboard score. One L4.

**C. Roofline at steady state with an image axis.** `configs/vl_batch_d1.toml` (`benchmarks/vl_batch.py`):
d1-3B, the same ImajevBench rows in three arms: the picture, text filler of the picture's LM token count, and
no picture. Pictures at 384, 768 and 1024 px (3:2), B = 1..64, in a packed batched path that d1's engine lacks
(`backends/d1.py`: one SigLIP2 call per batch, then one packed tree over every row, no padding). It is checked
against the engine's single-request answers. The cost model charges the vision tower and the projector
separately (`flops.vision_stacks`). Each series is compared at its own steady-state batch (kernel_busy >= 0.95,
not beaten by 5%). Runs on an L4 and an H100.

Smoke: `configs/mm_smoke.toml` (20 ImajevBench rows, `vl_batch` at one size with the check).

Prediction, written before the runs: an image arm is *more* compute-bound than its text twin. The SigLIP2 tower
is dense matmuls and all-pairs attention over about 1k patches per tile, so it raises FLOP/byte. The picture's
memory cost is mostly the 0.85 GB of vision weights, plus the vision activations at large batches.

## Findings (runs of 2026-11-20, code 14f7eda)

Short answer: a picture does make d1-3B's state much bigger and its decisions much more expensive per
decision, and it uses more activation memory per decision. But it does not push the model toward
memory-bound. With or without a picture, every steady-state point is compute-bound, far right of the ridge on
both GPUs, and the picture moves it further right. The accuracy question is not settled yet (see A).

### B. d1-3B on ImajevBench (L4, `results/20261120-142445-7427749f`, `-142920-33a46b0b`)

| 230 answerable rows | with pictures | pictures removed |
|---|---|---|
| accuracy | **71.3%** | 46.5% |
| image-only rows (101) / image + text rows (96) | 78.2% / 63.5% | 31.7% / 53.1% |
| text-only rows (33), same input both times | 72.7% | 72.7% |
| ECE / Brier | 0.100 / 0.389 | 0.078 / 0.585 |
| prompt tokens per decision, image rows | ~1,855 | ~65 |
| latency per decision (batch 1, engine path) | 429 ms | 35 ms |
| peak memory above the weights (5.98 GB) | 218 MB | 22 MB |

There are 0 invalid answers. The pictures carry the answers: removing them costs 25 points, close to the d1
card's own ablation (45.1 without images on its 11 benchmarks). One ImajevBench picture (1536x1024, capped to
1 MP) is about 1,790 LM tokens, about 30x the text it accompanies. So the "bigger state" premise holds for
real image decisions, by more than an order of magnitude.

### C. Steady state with an image axis (`results/20261120-141226-e1dd7b35` L4, `-141037-d11b3b27` H100)

Image tokens per picture: 96 at 384 px, 247 at 768 px, **1,783 at 1024 px**. The jump at 1024 px is the
processor switching from one tile to 512 px tiling plus a thumbnail. Each series is compared at its own B*.
"text" carries filler of the picture's LM token count; "none" is the row alone (~60 tokens).

| GPU | arm @ px | B* | ms / decision | TFLOP/s | FLOP/byte | kernel_busy | vision share | peak MB / decision |
|---|---|---|---|---|---|---|---|---|
| L4 | image 384 | 16 | 28.4 | 40.7 | 2,360 | 0.95 | 0.32 | 15.3 |
| L4 | text 384 | 16 | 19.3 | 41.6 | 2,121 | 0.99 | | 12.6 |
| L4 | image 1024 | 1 | 452.5 | 36.0 | 2,090 | 0.97 | 0.47 | 216.1 |
| L4 | text 1024 | 1 | 231.2 | 39.7 | 1,558 | 0.98 | | 144.5 |
| L4 | none | 32 | 7.7 | 41.0 | 1,698 | 0.98 | | 5.0 |
| H100 | image 384 | 64* | 3.24 | 388.9 | 6,216 | 0.87 | 0.30 | 15.2 |
| H100 | text 384 | 64 | 2.18 | 390.7 | 6,770 | 0.97 | | 12.5 |
| H100 | image 1024 | 32* | 45.8 | 455.5 | 12,030 | 0.91 | 0.43 | 216.1 |
| H100 | text 1024 | 4 | 25.3 | 375.9 | 5,219 | 1.00 | | 143.0 |
| H100 | none | 64 | 0.88 | 365.2 | 3,166 | 0.96 | | 4.9 |

\* The image arm never reached kernel_busy 0.95 on the H100 (0.87-0.91 at every B). B* is then the fastest
point, per the rule's fallback. On the L4 every series reached it. 768 px is in the records and sits between
the two. On the L4, the 1024 px image arm ran out of memory at B = 64 (recorded as `_oom`).

- **Cost of the modality at equal LM length:** image / text = **1.4-2.0x** time per decision (headline
  `vl_batch_image_text_ms`: 1.96 L4, 1.81 H100, at 1024 px). The SigLIP2 tower and projector take 30-47% of
  GPU time, rising with picture size.
- **Cost of a picture per decision:** image / none = **52x (H100) to 59x (L4)** at 1024 px. Almost all of it
  is the ~1.8k tokens the picture adds, plus the tower.
- **Memory:** at equal LM length the image arm peaks at **1.5x** the text arm's activation memory per decision
  (`vl_batch_image_text_mem` 1.50 / 1.51), from the vision tower's activations. Against no picture it is 43x
  (216 vs 5 MB per decision at 1024 px). It is capacity, not bandwidth: the L4's only OOM was the 1024 px image
  arm at B = 64. There is still no KV cache: d1 does one prefill pass and caches nothing.
- **Roofline:** measured ridges are 245 (Modal L4) and 227 (H100) FLOP/byte (`results/roofline_findings.md`).
  Every steady-state point is 1,558-12,030, so **all arms are compute-bound**. The picture raises FLOP/byte
  further (H100 1024 px: 12,030 vs 5,219 for the same LM length), as predicted: bidirectional attention over
  ~1k patches per tile, plus dense matmuls. So the modality does not move d1-3B toward memory-bound.
  Achieved FLOP/s at B* is 36-41 TFLOP/s on the L4 (63-73% of its measured bf16 peak, ~57 TFLOP/s) and 365-456 on the H100.
  The H100's best is the image arm at 1024 px, the most compute-dense work here. Its weaker point is
  utilization: the image path leaves 9-13% of wall time idle. That is likely the host-to-device copy of
  pixel_values (not pinned) and the vision-feature scatter between the two spans; it is not profiled yet.
- **Steady-state batch:** on the L4, image and text arms reach steady state at the same B (16 at 384 px, 8 at
  768, 1 at 1024): a big state saturates the small GPU by itself. On the H100 the image arm needs B = 32-64
  and still idles, while text at 1024 px saturates at B = 4.
- **Check:** the packed batched path matches the engine's own single-request answers within 0.027 (H100) and
  0.031 (L4) in probability (bf16 kernels differ by batch shape).

### A. Same capacity, wider data (L4, `results/20261120-141038-47e8326c`)

LFM2.5-VL-3B zero-shot: **40.8%** on the JEVal smoke's 200 questions (49 invalid: the 16-option cap, as for
every SemIf control), 57.6% on ImajevBench's text track. Its weights take 5.97 GB on the GPU, against 5.15 GB
for the text twin (+0.81 GB idle vision tower).

The text twin's first run (`20261120-141138-322c83e3`, not committed) is **invalid**. LFM2.5-2.6B turned out to
be a pure reasoning model: its chat template ends every prompt in an open `<think>` and ignores
enable_thinking. So the readout scored option letters at the start of a thought (20.4%, below always-yes on
yes/no questions). Fixed in 030ca10 (`semif_close_think`, an empty thought, on for that spec only). The
rerun, `configs/mm_text_twin_l4.toml`, is pending approval. Even with it, the pair differs in more than
image data: the twin was post-trained for reasoning, the VL model was not.

### Notes on the keys

- `vl_batch_*_padding` is negative: rows of a batch share their common prompt start (the trunk), which the cost
  model charges once, while `real_tokens` counts every row in full. Read it as prefix sharing, not padding.
- FLOP/byte counts weights once per batch, so it grows with B. Compare arms at the same B, or each at its B*.
