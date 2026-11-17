# QNEXT_* environment switches

Every engine optimization is behind an environment switch, and all of them are off by default. Without any `QNEXT_*`
variable set, the engine behaves like the llama.cpp code it is based on. The switches are read once, when the model
context is created (instrumentation switches are read the first time they are needed).

Categories:

- **profile**: part of the recommended configuration (see the profile below).
- **tuning**: a parameter of a profile switch. The default is the accepted value.
- **verification**: a debug or negative-control mode used to check correctness. Never set it for production use.
- **instrumentation**: logging or timing only. It may slow execution down (some modes synchronize the GPU) but does
  not change results.

## Recommended profile

Flags: `--ubatch-size 2048 --moe-cache-mib auto --checkpoint-min-step 1024 --backend-sampling --spec-type draft-mtp
--spec-draft-n-max 2 --spec-draft-cpu-moe --spec-draft-n-ubatch 512 -td 8 -tbd 8`, plus the draft model.

Environment:

```
QNEXT_MOE_BIGBATCH_BYPASS=1 QNEXT_MOE_BATCH_COPY=1 QNEXT_ELASTIC_KV=3 QNEXT_PF_OVERLAP=1 QNEXT_PF_OVERLAP_SLOTS=3
QNEXT_IDS_ASYNC=1 QNEXT_MTP_PRUNE=1 QNEXT_MTP_CROP=1 QNEXT_ELASTIC_COMPUTE=1 QNEXT_MOE_LOOKAHEAD=4
QNEXT_MOE_LOOKAHEAD_MAX_MIB=24 QNEXT_MOE_LOOKAHEAD_RANK=1 QNEXT_MOE_LOOKAHEAD_ISSUE=post QNEXT_MOE_LOOKAHEAD_P4_FAST=1
QNEXT_MOE_MISS_OVERLAP=1
```

## MoE expert cache and expert uploads

| Switch | Values (default) | What it does | Category |
|---|---|---|---|
| `QNEXT_MOE_BIGBATCH_BYPASS` | `1` (off) | Large ubatches (more than 32 tokens) upload the experts they need from host memory and do not read the MoE cache, so they leave the cache contents and its LRU state untouched. | profile |
| `QNEXT_MOE_BATCH_COPY` | `1` (off) | The missing experts of one layer are uploaded with a single batched copy call (`cudaMemcpyBatchAsync`) instead of one call per run of experts. | profile |
| `QNEXT_IDS_ASYNC` | `1` (off) | The selected expert ids of the next cached layer are read back into pinned memory right after the split that computes them is launched, so the MoE cache copy callback only waits on an event; slot maps are uploaded from pinned staging. Required by `QNEXT_MOE_LOOKAHEAD`. | profile |
| `QNEXT_MOE_LOOKAHEAD` | `4` (off) | For ubatches of 2 to 4 tokens, predicts the experts of the next layer from the residual stream before the current layer's router runs, and uploads the predicted, not yet cached experts on a second stream while the current layer computes; the compute stream waits on the upload only for slots it actually uses. `4` is the only accepted value. | profile |
| `QNEXT_MOE_LOOKAHEAD_MAX_MIB` | MiB (9 for 1-token, 14 for multi-token ubatches) | Upload budget per layer for the lookahead prefetch. The profile uses 24. | profile / tuning |
| `QNEXT_MOE_LOOKAHEAD_RANK` | `1` | Accepted for compatibility with the profile; ranked predictions (ordered by router score) are the only behaviour. Other values are ignored with a warning. | profile |
| `QNEXT_MOE_LOOKAHEAD_ISSUE` | `post` | Accepted for compatibility with the profile; the lookahead upload is always planned and issued right after the split launch. Other values are ignored with a warning. | profile |
| `QNEXT_MOE_LOOKAHEAD_P4_FAST` | `1` | Accepted for compatibility with the profile; the predictor always uses the cheaper fused mixer when the fused HC operator is available. Other values are ignored with a warning. | profile |
| `QNEXT_MOE_LOOKAHEAD_K` | 1..24 (10) | Predicted experts per token. | tuning |
| `QNEXT_MOE_LOOKAHEAD_MIN_LAYER` | >= 1 (2) | First layer that gets a prediction. | tuning |
| `QNEXT_MOE_LOOKAHEAD_MIN_TOK` | 1..4 (2) | Smallest ubatch that gets a prediction; the default keeps single-token decoding (no draft) unchanged. Values above 4 turn the lookahead off with a warning. | tuning |
| `QNEXT_MOE_LOOKAHEAD_PROTECT` | `0`/`1` (1) | With 1, slots holding predicted experts that are already cached are never chosen as victims for the same upload batch. | tuning |
| `QNEXT_MOE_MISS_OVERLAP` | `1` (off) | For ubatches of 2 to 8 tokens, a one-thread kernel publishes each layer's selected expert ids to mapped pinned memory right after the router; the next layer's predictor and the shared expert are moved after that publish, so the host reads the ids without synchronizing the compute stream, uploads the missed experts on the lookahead's upload stream while that tail runs, and the compute stream waits on the upload's event before the layer. Needs `QNEXT_MOE_LOOKAHEAD`, `QNEXT_IDS_ASYNC` and `QNEXT_MOE_BATCH_COPY`; refused (one log line, switch off) with an eval callback, a CPU cache share, `QNEXT_MOE_TIMING=2`, `GGML_CUDA_GRAPH_OPT=1` or `QNEXT_MOE_LOOKAHEAD_DEBUG_DEFER`; any value other than 0 or 1 is refused. If the publish counter runs ahead of the host's count, or a publish does not arrive within 2 s, it turns itself off for the rest of the context and the synchronized path takes over; a published id list whose length differs from what the lookup reads (not possible by construction) stops the process. Tuning: `QNEXT_MOE_MISS_OVERLAP_MIN_TOK` (1..8, 2). Verification: `QNEXT_MOE_MISS_OVERLAP_DEBUG=1` also reads the ids back the old way and counts mismatches (uses the old ids); `QNEXT_MOE_MISS_OVERLAP_DEBUG_SPIN_US=N` fills the miss slots with 0xFF and delays the upload stream by N us before each miss batch; `QNEXT_MOE_MISS_OVERLAP_DEBUG_SKIP_WAIT=1` (needs SPIN_US) is the negative control in which the compute stream does not wait, so output is expected to be wrong. | profile |
| `QNEXT_PF_OVERLAP` | `1` (off) | Ubatches of more than 32 tokens upload whole host expert tensors into a ring of device slots on a second stream, ahead of the layer that uses them, so uploads overlap compute. With `QNEXT_ELASTIC_KV=3` the ring borrows MoE cache bank memory for the duration of the graph. | profile |
| `QNEXT_PF_OVERLAP_SLOTS` | 2..8 (2) | Number of ring slots for `QNEXT_PF_OVERLAP`. The profile uses 3. | profile / tuning |

## Memory: elastic KV cache and compute buffer

| Switch | Values (default) | What it does | Category |
|---|---|---|---|
| `QNEXT_ELASTIC_KV` | `0`..`3` (0) | KV and indexer caches on CUDA virtual memory. 1: elastic buffers, fully mapped. 2: only the cells in use plus a margin are mapped; the rest is parked. 3: as 2, and the parked memory is lent to the MoE cache as extra expert slots, taken back when the context grows. | profile |
| `QNEXT_ELASTIC_UNIT` | cells (4096) | Growth unit of the elastic KV cache, rounded up to a multiple of 256. | tuning |
| `QNEXT_ELASTIC_COMPUTE` | `1` (off) | The target context's CUDA compute buffer is an elastic buffer; while only small graphs run, the part above their high-water mark is lent to the MoE cache, and it is taken back before a larger graph runs. Needs `QNEXT_ELASTIC_KV=3`. | profile |
| `QNEXT_ELASTIC_COMPUTE_CHUNK_MIB` | 2..1024 (16) | Size of one mapping of the elastic compute buffer; lending moves whole mappings. | tuning |
| `QNEXT_ELASTIC_COMPUTE_SLACK_MIB` | 0..4096 (16) | Margin added to the measured high-water mark before the rest is lent. | tuning |
| `QNEXT_ELASTIC_COMPUTE_VA_MIB` | MiB (3072) | Address space reserved in the MoE cache banks for lent compute memory. | tuning |

## Speculative decoding draft (MTP) context

| Switch | Values (default) | What it does | Category |
|---|---|---|---|
| `QNEXT_MTP_PRUNE` | `1` (off) | A draft ubatch with no outputs (prompt pass, verify catch-up) is pruned to the nodes that write the draft's persistent state (K/V and indexer rows); attention output, FFN, head and sampling are skipped. | profile |
| `QNEXT_MTP_CROP` | `1` (off) | The draft graph keeps only its output rows after the attention part, so the FFN/MoE and head run on the output rows only, and with `--spec-draft-cpu-moe` the experts stay on the CPU without a device copy. | profile |

## Verification

| Switch | Values (default) | What it does | Category |
|---|---|---|---|
| `QNEXT_MOE_LOOKAHEAD_DEBUG` | `1` (off) | Lookahead victim slots are filled with 0xFF before the upload and the upload stream is delayed, so a missing compute-stream wait shows up as wrong output. | verification |
| `QNEXT_MOE_LOOKAHEAD_DEBUG_SPIN_US` | 0..100000 us (200) | Length of the upload-stream delay in debug mode; 0 disables the delay. | verification |
| `QNEXT_MOE_LOOKAHEAD_DEBUG_DEFER` | `1` (off, needs DEBUG) | The host holds back each upload batch until two copy callbacks later, or until a compute-stream wait needs it, so the test does not depend on how the GPU orders the two streams. | verification |
| `QNEXT_MOE_LOOKAHEAD_DEBUG_SKIP_WAIT` | `1` (off, needs DEBUG) | Negative control: the compute stream does not wait for in-flight lookahead uploads. Output is expected to be wrong (with DEBUG_DEFER it fails reliably), which proves the wait is what keeps the lookahead correct. | verification |
| `QNEXT_RING_POISON` | `0`..`2` (0) | Overlap ring in the MoE cache banks: 1 fills the ring range with 0xFF at each return before it is cleared; 2 fills it and skips the clearing (reproduces the read-past-the-end hazard the clearing prevents). | verification |
| `QNEXT_MTP_STATE_HASH` | N > 0 (off) | Logs a 64-bit hash of the draft context's K, V and indexer rows at prompt/draft transitions and every N draft ubatches (synchronizes). | verification |
| `QNEXT_COMPUTE_FILL` | byte 0..255 (off) | Fills the scheduler's compute buffers with the byte when they are (re)allocated; a result that changes with the byte means a kernel reads memory it did not write. | verification |
| `QNEXT_COMPUTE_FILL_CTX` | `draft`, `target`, `both` (both) | Context(s) that `QNEXT_COMPUTE_FILL` applies to. | verification |
| `QNEXT_COMPUTE_FILL_EVERY` | `1` (off) | Fill before every newly allocated graph instead of only on (re)allocation. | verification |
| `QNEXT_POOL_FILL` | byte 0..255 (off) | Fills every CUDA memory pool allocation with the byte; detects kernels that read pool scratch they did not write. | verification |

## Instrumentation

| Switch | Values (default) | What it does | Category |
|---|---|---|---|
| `QNEXT_MOE_STATS` | N > 0 (off) | Logs MoE cache statistics (hit rate, uploads, batched copies, async ids, lookahead, elastic memory) every N rounds, and a line of large-batch copy statistics per graph. | instrumentation |
| `QNEXT_MOE_TIMING` | `0`..`2` (0, needs `QNEXT_MOE_STATS`) | Host time of the MoE cache copy phases and of the graph evaluations; mode 2 also synchronizes after the miss uploads, which perturbs execution. | instrumentation |
| `QNEXT_STEP_EVENTS` | N > 0 (off) | Times 1 of every N small-batch evaluations with CUDA events around the split launches, the miss uploads and the lookahead uploads and waits. | instrumentation |
| `QNEXT_STEP_EVENTS_EVERY` | rounds (100) | Log interval of `QNEXT_STEP_EVENTS`. | instrumentation |
| `QNEXT_OP_TIMING` | `1` (off) | Per-operator GPU time for decode-class graphs (at most one token); runs those graphs without CUDA graphs and synchronizes after every node, so the totals are a diagnostic, not decode performance. | instrumentation |
| `QNEXT_OP_TIMING_EVERY` | steps (200) | Report interval of `QNEXT_OP_TIMING`. | instrumentation |

The `qnext_moe_lookahead: C19: predictor tensors registered in the last graph = N (layers registered twice = D)` log
line is printed when lookahead is on; D must be 0 (one predictor per layer).
