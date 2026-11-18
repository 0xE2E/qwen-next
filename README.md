# qwen-next

An inference engine specialized for **one model on one machine**: Qwen3.8-Flash-Next (GGUF `UD-IQ3_XXS`, 83.6 GB, a Mixture-of-Experts model with 512 routed experts per block) on a laptop with an RTX 5080 Laptop GPU (16 GB), a Core Ultra 9 275HX and 128 GB RAM, running Windows 11.

It starts from [llama.cpp](https://github.com/ggml-org/llama.cpp) as the seed, so it never ran slower than llama.cpp did on this machine, and then specializes it in measured cycles. The seed is the llama.cpp source published by Unsloth (`unslothai/llama.cpp`, source commit `b8645dd`), which already contains the model's graph, a GPU expert cache and MTP speculative decoding.

Every accepted engine change is meant to be **lossless**: with greedy decoding, each change was kept only if the output tokens matched the same configuration without it. One exception is on record: a draft-buffer change altered the tokens of one 20-token test request and was accepted with that caveat. Output can also differ from the stock configuration, because moving expert math between CPU and GPU changes rounding.

Speculative decoding is a separate matter. With a drafter, every emitted token is still the target model's own greedy choice, but a multi-token verify batch rounds differently from a single-token step, so the text can diverge from the no-draft run where two candidate tokens are nearly tied.

## Results

Measured on the machine above at a 90,000-token context window. Tokens per second, greedy decoding, one user.

| Configuration | Prefill | Decode |
|---|---|---|
| llama.cpp seed, original flags (expert cache off, no draft) | ~800 | ~30 |
| llama.cpp seed, its own flags tuned (`--moe-cache-mib auto --ubatch-size 2048`), no code changes | ~845 | ~44 |
| qwen-next, MTP draft block, 2 draft tokens (`-Mtp`) | ~1,480 | ~80 |
| **qwen-next, default profile: DFlash 2 drafter, 2 draft tokens** | **~1,480** | **~83** |
| qwen-next without a draft model (`-NoDraft`) | ~1,450 | ~50 |

The rows above use one fixed test request (a 1,808-token prompt). On five replayed coding-agent sessions (tool calls and reasoning included, up to 400 generated tokens, GPU cooled to near idle before each decode), decode speed by how full the 90k window is:

| Configuration | ~2k tokens | ~42k tokens | ~84k tokens |
|---|---|---|---|
| **DFlash 2 drafter, 2 draft tokens (default)** | **75.0** | **66.5** | **57.0** |
| MTP draft block, 2 draft tokens | 73.2 | 58.3 | 43.7 |
| No draft | 48.8 | 46.2 | 45.3 |
| DFlash 2 drafter, 7 draft tokens | 57.0 | 52.3 | 35.9 |

Long drafts lose on this model: each extra verified token routes to experts the cache does not hold (cache hit rate 71% at 7 draft tokens against 88% at 2), and fewer drafted tokens are accepted.

Read these with care:

- One model, one quantization, one machine, one operating system. Nothing here is a general benchmark.
- Rows were measured in separate sessions; run-to-run drift of 5–10% is normal on this laptop. The session-replay table is one run per configuration, five prompts per cell.
- "Same tokens" was checked with greedy decoding on a fixed set of requests (short prompts of every batch-size class, 12k and 60k prompts, a full-context fill, repeated and cached requests). It is a test result, not a proof.
- MTP and DFlash speculative decoding are llama.cpp features, and the DFlash 2 drafter is a third-party model ([incoai/Qwen3.8-Flash-Next-DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2)). The work here was making speculation pay off inside a 16 GB budget.

## What is different from the seed

All switches are off by default; the starter script turns on the accepted set.

| Area | Change | Switch |
|---|---|---|
| Prefill | Large batches bypass the expert cache and stream experts once | `QNEXT_MOE_BIGBATCH_BYPASS` |
| Transfers | Missed experts move in a few large asynchronous copies | `QNEXT_MOE_BATCH_COPY` |
| Transfers | Next layer's uploads overlap the current layer's compute during prefill, on slots borrowed from the cache | `QNEXT_PF_OVERLAP`, `QNEXT_PF_OVERLAP_SLOTS` |
| Memory | The KV cache's worst-case reservation is backed by GPU virtual memory and lent to the expert cache until the context needs it | `QNEXT_ELASTIC_KV` |
| Memory | The compute-buffer reservation is lent to the expert cache during decode | `QNEXT_ELASTIC_COMPUTE` |
| Host round trips | Routing ids are read back without a full stream sync | `QNEXT_IDS_ASYNC` |
| Draft | The MTP draft pass computes only what it reads; its buffers are cropped and the memory goes to the cache | `QNEXT_MTP_PRUNE`, `QNEXT_MTP_CROP` |
| Prefetch | While layer L computes, a cheap probe predicts layer L+1's experts and uploads them on a second stream; compute waits on the GPU only if a selected expert is still in flight | `QNEXT_MOE_LOOKAHEAD*` |
| Transfers | Each layer's routing ids are published from inside the previous layer's graph, so missed experts upload while the rest of that graph still runs; compute waits on the upload's event | `QNEXT_MOE_MISS_OVERLAP` |
| Sampling | Greedy sampling stays on the GPU | `--backend-sampling` |

Things that were built, measured and shelved are part of the record too: for example a bit-exact kernel fusion that removed about 9% of the kernels per step and changed speed by 0%.

## Quick start

1. Build the engine: see [docs/BUILD.md](docs/BUILD.md).
2. Get the model: the `UD-IQ3_XXS` GGUF shards of Qwen3.8-Flash-Next and, for speculative decoding, the DFlash 2 drafter GGUF (`Q4_K_M` is enough; the 8-bit file accepted no more drafts here). The MTP draft GGUF is optional.
3. Start the server:

```powershell
.\start-qwen-next.ps1 -Model <path to first GGUF shard> -DFlash <path to DFlash 2 drafter GGUF>
```

The script prints the exact command line and environment it uses. Useful options: `-Mtp -Draft <MTP draft GGUF>` (draft with the model's own MTP block), `-NoDraft`, `-Context 64000`, `-Port 8081`, `-Stock` (the seed's original flags, for comparison), `-DryRun`. The server speaks the usual llama.cpp HTTP API on `http://127.0.0.1:8080`.

Requirements: a CUDA GPU with about 15 GB of usable VRAM, enough RAM to hold the model (128 GB here), and Windows with MSVC and the CUDA toolkit to build. Other GPUs and Linux may work but were never tested; the tuned numbers (cache sizes, prefetch budget, thread counts) are specific to this machine.

## Repository layout

- `engine/` — the engine: llama.cpp source with the qwen-next changes (MIT, see `engine/LICENSE`).
- `start-qwen-next.ps1` — starter with the accepted profile.
- `docs/` — design and contract documents, build instructions.
- `tools/` — generic utilities (routing-trace cache simulator, hardware probes in `tools/probes/`).

## Status

Research code for one machine. The engine switches of the accepted profile passed their identity and stress gates on that machine with the MTP draft; the DFlash 2 default was chosen on the session-replay speed runs above; some default-off diagnostic paths are unreviewed. Use it to learn from, not as a supported product.
