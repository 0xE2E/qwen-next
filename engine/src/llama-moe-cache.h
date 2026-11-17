#pragma once

#include "ggml-backend.h"

#include <map>
#include <memory>
#include <vector>

struct llama_model;
class llama_elastic_lender;

// keeps recently and repeatedly used experts of host-resident MoE layers in a device buffer
// each layer has a slot map in host memory: when the scheduler copies it to the device, the copy callback uploads the missing experts
class llama_moe_cache {
public:
    // the largest ubatch (tokens) that reads its experts through the cache slot maps; larger ubatches use the scheduler's
    // expert copies (and QNEXT_PF_OVERLAP's ring for exactly these ubatches): the one definition of "large batch"
    static constexpr int64_t max_batch = 32;

    // backends are all the backends of the context, each GPU gets its own cache of the given size for the layers assigned to it
    // cpu_share is the share of the misses computed on the CPU instead of uploaded
    // elastic_layout (QNEXT_ELASTIC_KV=3, the same in the --fit pass): banks on elastic memory; elastic_lend_bytes: the KV chunks
    // the banks may borrow on top of `size`, see llama-elastic.h
    llama_moe_cache(const llama_model & model, bool no_alloc, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, float cpu_share, bool elastic_layout = false, size_t elastic_lend_bytes = 0);
    ~llama_moe_cache();

    // QNEXT_ELASTIC_KV=3: the banks are elastic and can take lent KV chunks, nullptr otherwise
    llama_elastic_lender * elastic_lender();
    // the banks are on elastic memory (QNEXT_ELASTIC_KV=3), the same in the --fit pass
    bool elastic_layout() const;

    // QNEXT_PF_OVERLAP with elastic banks: the overlap ring lives in a fixed range of the own bank slots, lent only
    // for the duration of a large-batch graph. ring_plan: up to n_want ring slots of slot_size bytes, returns how many;
    // ring_ptr(k): their fixed device addresses in ring_buffer(); ring_borrow parks the range (returns the experts
    // dropped), ring_return clears the head of every ring-range slot on the compute stream (MMQ over-reads into the
    // next slot) and makes the slots active and empty again, returns the slots unparked; no KV growth or shrink in between
    int32_t ring_plan(size_t slot_size, int32_t n_want);
    ggml_backend_buffer_t ring_buffer() const;
    void *  ring_ptr(int32_t k) const;
    size_t  ring_borrow();
    int64_t ring_return();
    int64_t ring_slots_parked() const;

    // the device that caches layer il
    ggml_backend_t backend(int32_t il) const;

    // the slot map of layer il, if its experts can be read from the cache for n_tokens tokens, nullptr otherwise
    // cpu_part: the graph computes the misses in the CPU map of the layer (see get_cpu_map)
    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used, bool cpu_part) const;

    // the experts of w in the cache, nullptr if w is not cached
    ggml_tensor * get_experts(const ggml_tensor * w) const;

    // the experts of layer il computed on the CPU: I32 [1, n_expert] in host memory, e for these experts, -1 for the others
    // nullptr if no misses of layer il are computed on the CPU for ubatches of n_tokens tokens
    ggml_tensor * get_cpu_map(int32_t il, int64_t n_tokens) const;

    // ggml_backend_sched copy callback, returns false if src is not a slot map
    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph);

    // QNEXT_IDS_ASYNC=1 (default off), see llama-moe-cache.cpp: enabled (all preconditions met at init)
    bool ids_async_enabled() const;
    // from the scheduler's split callback, after a split was launched: enqueue the ids readback of the next split's lookup
    void ids_post_split(const ggml_backend_sched_split_info * info);
    // before a graph evaluation: forget pending readbacks
    void ids_async_reset();

    // QNEXT_STEP_EVENTS: called by copy() right before (which = 0) and after (which = 1) the miss uploads are enqueued
    // on the compute stream; with QNEXT_MOE_MISS_OVERLAP which = 2 / 3 and backend is the upload stream's backend
    void set_copy_mark_hook(void (*fn)(void * ud, ggml_backend_t backend, int which), void * ud);

    // QNEXT_MOE_MISS_OVERLAP=1 (default off), see llama-moe-cache.cpp
    // the graph publishes the selected ids of layer il (n_tokens rows) right after its router and expands the work that
    // does not depend on its experts before its slot lookup; a pure function of the configuration, the cache layout and
    // n_tokens (false with the switch unset)
    bool miss_overlap(int32_t il, int64_t n_tokens) const;
    // the GGML_OP_QNEXT_PUBLISH node of layer il (kind 0: the selected ids, 1: the predicted ids of layer il)
    ggml_tensor * miss_overlap_publish(ggml_context * ctx, ggml_tensor * ids, int32_t il, int kind) const;
    // the runtime part is active (llama_context then installs the scheduler's no-sync callback)
    bool miss_overlap_enabled() const;
    // for the scheduler: the copy callback of this slot map orders itself (a publish of its ids is pending)
    bool copy_nosync(const ggml_tensor * src) const;
    // turn the switch off before any graph is built (reason logged), e.g. when an eval callback is set
    void miss_overlap_refuse(const char * reason);

    // for large batches: copy the experts of w that are in the cache, starting at expert e and up to expert last, to the copy dst of w
    // returns the number of experts copied, 0 if expert e is not in the cache
    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last);

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

    // QNEXT_MOE_TIMING (0 = off, 1, 2), only active together with QNEXT_MOE_STATS=N, see llama-moe-cache.cpp
    int  timing_mode() const;
    void timing_graph_begin();         // before a graph evaluation
    bool timing_graph_used() const;    // the graph just evaluated went through the cache with a slot map (a small batch)
    void timing_graph_end(int64_t ns); // host wall time of that graph evaluation, including the GPU sync added in timing mode

    // QNEXT_MOE_LOOKAHEAD: before a graph evaluation, the I32 [lookahead_k(), n_rows] predicted ids of layer il, a graph
    // output computed before the slot lookup of layer il - 1; they are read back together with the real ids of layer il - 1
    void prefetch_set_ids(int32_t il, ggml_tensor * ids);
    void prefetch_clear(); // after the evaluation (the copy callbacks have run), forget the ids of the graph
    bool prefetch_ids_wanted() const; // QNEXT_MOE_LOOKAHEAD keeps predicted ids (prefetch_set_ids)

    // QNEXT_MOE_LOOKAHEAD=4 (default off), see llama-moe-cache.cpp: cross-layer predicted lookahead prefetch on a second
    // stream for ubatches of up to 4 tokens
    // the predictor (4) the graph builds during layer il - 1 for layer il, 0 if none; a pure function of the
    // configuration, the cache layout and n_tokens (reserve, --fit and keep-region probe graphs get the same nodes)
    int  lookahead_mode(int32_t il, int64_t n_tokens) const;
    int  lookahead_k() const; // predicted experts per token: the ne[0] of the I32 ids of a predictor
    // before a ubatch of n_tokens tokens: drain if it may take the MMQ path (more tokens than the smallest MMVQ batch
    // limit of the cached expert types, at most 8), see lookahead_drain
    void lookahead_drain_ubatch(int64_t n_tokens);
    // finish all lookahead uploads (synchronize the upload stream) and clear the in-flight and prefetched-unused state;
    // reason is counted in the qnext_moe_lookahead line; a no-op (not counted) if nothing is in flight or unused
    void lookahead_drain(const char * reason);

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using llama_moe_cache_ptr = std::unique_ptr<llama_moe_cache>;
