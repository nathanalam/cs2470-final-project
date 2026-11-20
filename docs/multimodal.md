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

## Findings

Pending the runs.
