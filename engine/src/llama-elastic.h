#pragma once

#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// QNEXT_ELASTIC_KV (default off): the KV and indexer caches on CUDA virtual memory ("elastic" buffers)
//   unset/0: the plain CUDA buffers, as before
//   1: the KV/indexer buffers are elastic buffers, fully mapped at load; nothing else changes
//   2: only the cells in use plus a margin are mapped in the KV/indexer buffers; the physical chunks of the other cells
//      stay mapped in a private parking ("spare") range of the same cache, so the VRAM use is unchanged, and move back
//      (zero filled) before any cell beyond the mapped ones is read or written
//   3: as 2, and the parked chunks are lent to the MoE expert cache at load (extra slots at the tails of its banks, see
//      llama-moe-cache.cpp); a KV growth takes chunks back from the bank tails (their slots are parked, the experts in
//      them dropped), a KV shrink at decode entry (hysteresis: at least 2 units) gives them back. Refused (level 2 with a
//      warning) with a cache CPU share, LLAMA_MOE_CACHE_TRACE, no MoE cache, or more than one GPU
// QNEXT_ELASTIC_UNIT=N: the growth unit in cells (default 4096, rounded up to a multiple of 256, at least 256)
// with QNEXT_MOE_STATS=N: one "qnext_elastic:" line per cache every N applied ubatches, besides the per-event lines
// supported only for the qwen4exp target context (not the MTP draft), with flash attention (V not transposed), one KV
// stream, the KV offloaded to CUDA devices with VMM; otherwise a warning and the plain buffers

// the level asked by the environment (0..3), read once
int llama_elastic_env_level();

// the request is a property of the memory being created: llama_context sets it around create_memory for the contexts
// it supports, and llama_kv_cache reads it in its constructor (thread local, so concurrent loads do not interfere)
struct llama_elastic_request_scope {
    explicit llama_elastic_request_scope(int level);
    ~llama_elastic_request_scope();
private:
    int prev;
};

int llama_elastic_requested();

// the elastic buffer type of the device of a plain CUDA device buffer type, nullptr if not available
ggml_backend_buffer_type_t llama_elastic_buft_of(ggml_backend_buffer_type_t buft);

// the elastic helpers of the CUDA backend (proc addresses of its registry); ok() is false if not available
struct llama_elastic_fns {
    ggml_backend_buffer_type_t (*buft_of)(ggml_backend_buffer_type_t) = nullptr;
    ggml_backend_buffer_t (*reserve)(ggml_backend_buffer_type_t, size_t) = nullptr;
    bool   (*is_elastic)(ggml_backend_buffer_t) = nullptr;
    size_t (*granularity)(ggml_backend_buffer_t) = nullptr;
    size_t (*buft_granularity)(ggml_backend_buffer_type_t) = nullptr;
    size_t (*n_chunks)(ggml_backend_buffer_t) = nullptr;
    size_t (*n_mapped)(ggml_backend_buffer_t) = nullptr;
    bool   (*chunk_mapped)(ggml_backend_buffer_t, size_t) = nullptr;
    bool   (*map_new)(ggml_backend_buffer_t, const size_t *, size_t) = nullptr;
    void   (*sync)(ggml_backend_buffer_t) = nullptr;
    bool   (*move)(ggml_backend_buffer_t, const size_t *, ggml_backend_buffer_t, const size_t *, size_t, bool) = nullptr;
    // optional (see spans_ok): whole-mapping moves for QNEXT_ELASTIC_COMPUTE
    ggml_backend_buffer_type_t (*span_buft)(ggml_backend_buffer_type_t, size_t) = nullptr;
    size_t (*span_of)(ggml_backend_buffer_t, size_t) = nullptr;
    bool   (*move_spans)(ggml_backend_buffer_t, const size_t *, ggml_backend_buffer_t, const size_t *, size_t, bool, size_t *) = nullptr;
    bool resolved = false;

    bool ok() const;
    bool spans_ok() const { return resolved && span_buft && span_of && move_spans; }
};

const llama_elastic_fns & llama_elastic_fns_of(ggml_backend_dev_t dev);

class llama_kv_elastic;

// level 3: the MoE expert cache takes the parked KV chunks into its bank tails
class llama_elastic_lender {
public:
    virtual ~llama_elastic_lender() = default;
    // take up to n chunks of src (an elastic buffer of the cache's device) into the bank tails; returns how many were
    // taken: chunks[0, result) were moved, the others stay in src. The caller guarantees no GPU work uses them.
    virtual size_t give(ggml_backend_buffer_t src, const size_t * chunks, size_t n) = 0;
    // move n lent chunks back to the unmapped chunks `chunks` of dst, zero filled (synchronizes the device first,
    // parks the bank slots on them); aborts if it cannot
    virtual void take(ggml_backend_buffer_t dst, const size_t * chunks, size_t n) = 0;
    // for the stats line
    virtual std::string describe() const = 0;
    // a KV cache that lent to it; told when the lender goes away (lender_gone)
    virtual void attach(llama_kv_elastic * borrower) = 0;
    // chunks of other buffers mapped in the banks now (all lenders together), active slots beyond the own
    // ones, experts dropped from parked slots so far
    virtual size_t   lent_chunks() const = 0;
    virtual int64_t  lent_slots()  const = 0;
    virtual uint64_t dropped()     const = 0;

    // whole mappings ("spans" of one or more chunks, one physical handle each) on top of the bank tails, for
    // QNEXT_ELASTIC_COMPUTE. give_spans takes up to n spans of src (heads = their first chunks), returns how many
    // ([0, result) moved, zero filled); take_spans moves every span back to the heads of dst (synchronizes the device,
    // parks the slots on them). Spans always sit above the chunks lent by give/take: before give/take run while spans are
    // lent, the evict callback is called (its owner takes them back). n_calls (if given) += the driver calls made
    virtual size_t give_spans(ggml_backend_buffer_t src, const size_t * heads, size_t n, size_t * n_calls) = 0;
    virtual void   take_spans(ggml_backend_buffer_t dst, const size_t * heads, const size_t * lens, size_t n, size_t * n_calls) = 0;
    virtual size_t n_spans() const = 0;
    virtual void   set_span_evict(void (*fn)(void * ud), void * ud) = 0;
};

// QNEXT_ELASTIC_COMPUTE=1 (default off; needs QNEXT_ELASTIC_KV=3 and elastic MoE cache banks): the CUDA compute
// buffer of the qwen4exp target context is an elastic buffer (fully mapped, as a plain one); while only small graphs
// (<= llama_moe_cache::max_batch tokens) run, its chunks above the keep region [0, keep) + one guard chunk are lent to the
// MoE cache like parked KV chunks (extra bank slots, zero filled when they arrive); they come back (zero filled, the
// experts in them dropped) before any graph whose assignment reaches past the keep region is given data or computed
int llama_elastic_compute_env(); // 0/1, read once

class llama_compute_elastic {
public:
    // gran: the chunk size; slack: bytes added to a measured high-water mark before rounding to chunks
    llama_compute_elastic(llama_elastic_lender * lender, const llama_elastic_fns & api, size_t gran, size_t slack);
    ~llama_compute_elastic();

    // the keep region from the high-water mark of the worst small graph (measured at reserve time; replaces the earlier
    // value, never below the safety-net floor; takes back first if it grows into the lent region); what: for the log
    void set_keep(size_t high_water, const char * what);
    size_t keep() const { return keep_bytes; }
    size_t guard() const { return guard_chunks*gran; }

    bool lent() const { return n_lent > 0; }
    bool disabled() const { return off_reason != nullptr; }
    void disable(const char * why);

    // lend the chunks of buf (the compute buffer, an elastic buffer with one chunk) above keep + guard; synchronizes the
    // device first. Returns the chunks lent
    size_t give(ggml_backend_buffer_t buf);
    // take every lent chunk back into buf (the buffer they were lent from); `why` for the log
    void take(ggml_backend_buffer_t buf, const char * why);
    // after an allocation: if the assignment of buf reaches past the keep region while lent, take back first; a small
    // graph (n_tokens in [1, max_batch]) doing so raises the keep region (one warning). Returns true if it took back
    bool check_alloc(ggml_backend_buffer_t buf, size_t high_water, int64_t n_tokens, int64_t max_batch);

    void log(const char * event) const;

private:
    llama_elastic_lender * lender;
    const llama_elastic_fns & api;
    size_t gran;
    size_t slack;
    size_t keep_bytes = 0;     // the keep region [0, keep_bytes), a multiple of gran
    size_t keep_floor = 0;     // raised by the safety net, kept by later measurements
    size_t guard_chunks = 1;   // mapped above it, for reads past the end of a tensor

    ggml_backend_buffer_t lent_buf = nullptr; // the buffer the lent chunks belong to
    size_t lent_first = 0;     // the lowest lent chunk
    size_t n_lent     = 0;     // chunks lent
    size_t buf_chunks = 0;     // chunks of lent_buf
    std::vector<size_t> lent_heads; // the heads of the lent spans (mappings) of lent_buf
    std::vector<size_t> lent_lens;  // their chunk counts

    static void evict_cb(void * ud);
    uint64_t last_calls = 0;   // driver calls of the last transition
    uint64_t n_evict    = 0;   // take-backs asked by KV growth/shrink

    const char * off_reason = nullptr;

    // stats
    uint64_t n_give = 0;
    uint64_t n_take = 0;
    uint64_t n_net  = 0;       // take-backs forced by a small graph past the keep region
    double   ms_give = 0.0, ms_take = 0.0, last_ms = 0.0;
    uint64_t dropped_total = 0;
    uint64_t last_dropped  = 0;
    int64_t  last_slots    = 0; // slots gained by the last give
    uint64_t n_logged = 0;
    bool     warned_net = false;
};

// the mapping state of the K/V (or indexer) tensors of one cache
class llama_kv_elastic {
public:
    struct tensor_desc {
        size_t offset; // in the buffer
        size_t row;    // bytes per cell
        size_t size;   // bytes of the tensor (ggml_nbytes)
    };

    struct buffer_desc {
        ggml_backend_buffer_t    buf; // elastic, owned by the cache
        std::vector<tensor_desc> tensors;
    };

    // the buffers must be elastic and fully mapped; level >= 2 parks the chunks beyond `unit` cells
    llama_kv_elastic(const char * name, int level, uint32_t kv_size, std::vector<buffer_desc> bufs);
    ~llama_kv_elastic();

    // the cells [0, n) are about to be read or written: map them (and a margin) if they are not
    void need(uint32_t n);

    // every cell (K-shift rotates the whole cache)
    void need_all();

    // level 3, at decode entry: the cells [0, n) are in use or about to be; return the chunks past the margin to the
    // lender if at least 2 units can go
    void maybe_shrink(uint32_t n);

    uint32_t mapped_cells() const { return n_mapped; }

    // level 3: the parked chunks, and lending them all (as far as the lender takes them)
    size_t parked_bytes() const;
    void lend(llama_elastic_lender * lender);
    void lender_gone();

    // one applied ubatch, for the periodic stats line (QNEXT_MOE_STATS)
    void tick();

private:
    struct buf_state {
        buffer_desc d;
        ggml_backend_buffer_t spare = nullptr; // owned
        size_t n_spare = 0; // chunks of the spare range
        size_t top     = 0; // the spare chunks [0, top) are mapped (a stack)
        size_t lent    = 0; // chunks of this buffer now mapped in the lender
        size_t lost    = 0; // chunks that were lent when the lender went away: released with its banks, remapped fresh if needed
    };

    std::vector<uint8_t> want_of(const buf_state & bs, uint32_t cells) const;
    void map_to(uint32_t cells, const char * why);
    void shrink_to(uint32_t cells);
    void log(const char * event) const;
    uint32_t cells_backed() const; // cells whose chunks are mapped in every tensor (>= n_mapped)

    std::string name;
    int      level;
    uint32_t kv_size;
    uint32_t unit;
    uint32_t n_mapped; // the cells [0, n_mapped) are mapped in every tensor

    std::vector<buf_state> bufs;

    llama_elastic_lender * lender = nullptr;

    // stats
    uint64_t n_grow   = 0;
    uint64_t n_shrink = 0;
    double   last_ms  = 0.0;  // of the last event that moved or mapped chunks
    size_t   last_chunks = 0; // chunks moved by that event
    double   total_ms = 0.0;
    int64_t  stats_every = 0;
    int64_t  ticks = 0;
};
