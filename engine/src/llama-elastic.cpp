#include "llama-elastic.h"

#include "llama-impl.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

// QNEXT_ELASTIC_KV, see llama-elastic.h
//
// Invariant: every physical chunk created for a cache buffer at load stays mapped at all times, in exactly one place: the
// cache buffer itself, the cache's spare range, or (level 3) a bank tail of the MoE cache. A move unmaps a chunk and maps
// the same physical allocation elsewhere with nothing in between, so the VRAM in use never changes and a growth cannot run
// out of memory. Per cache buffer: chunks mapped in the buffer + top (spare) + lent == chunks of the buffer.
//
// Mapping rule: a cache buffer holds tensors whose cell i occupies the bytes [offset + i*row, offset + (i+1)*row) (one
// stream, V not transposed). "cells [0, c) mapped" means: every chunk overlapping [offset, offset + c*row) of every tensor
// is mapped; c == kv_size means every chunk of the buffer (the level 1 state). Adjacent tensors may share a chunk, which
// is then mapped as soon as either needs it.

bool llama_elastic_fns::ok() const {
    return buft_of && reserve && is_elastic && granularity && buft_granularity && n_chunks && n_mapped && chunk_mapped &&
        map_new && sync && move;
}

// the functions live in the CUDA backend, reached through its registry like the other QNEXT helpers
const llama_elastic_fns & llama_elastic_fns_of(ggml_backend_dev_t dev) {
    static llama_elastic_fns api;
    if (api.resolved || dev == nullptr) {
        return api;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (reg == nullptr) {
        return api;
    }
    auto get = [&](const char * name) { return ggml_backend_reg_get_proc_address(reg, name); };
    api.buft_of          = (decltype(api.buft_of))          get("ggml_backend_cuda_elastic_buffer_type_of");
    api.reserve          = (decltype(api.reserve))          get("ggml_backend_cuda_elastic_buffer_reserve");
    api.is_elastic       = (decltype(api.is_elastic))       get("ggml_backend_cuda_buffer_is_elastic");
    api.granularity      = (decltype(api.granularity))      get("ggml_backend_cuda_elastic_granularity");
    api.buft_granularity = (decltype(api.buft_granularity)) get("ggml_backend_cuda_elastic_buft_granularity");
    api.n_chunks         = (decltype(api.n_chunks))         get("ggml_backend_cuda_elastic_n_chunks");
    api.n_mapped         = (decltype(api.n_mapped))         get("ggml_backend_cuda_elastic_n_mapped");
    api.chunk_mapped     = (decltype(api.chunk_mapped))     get("ggml_backend_cuda_elastic_chunk_mapped");
    api.map_new          = (decltype(api.map_new))          get("ggml_backend_cuda_elastic_map_new");
    api.sync             = (decltype(api.sync))             get("ggml_backend_cuda_elastic_sync");
    api.move             = (decltype(api.move))             get("ggml_backend_cuda_elastic_move");
    api.span_buft        = (decltype(api.span_buft))        get("ggml_backend_cuda_elastic_span_buffer_type");
    api.span_of          = (decltype(api.span_of))          get("ggml_backend_cuda_elastic_span_of");
    api.move_spans       = (decltype(api.move_spans))       get("ggml_backend_cuda_elastic_move_spans");
    api.resolved         = api.ok();
    return api;
}

namespace {

const llama_elastic_fns & api_of_buf(ggml_backend_buffer_t buf) {
    return llama_elastic_fns_of(ggml_backend_buft_get_device(ggml_backend_buffer_get_type(buf)));
}

thread_local int g_request = 0;

constexpr uint32_t unit_default = 4096; // cells: 12 layers x (4 + 4 + 2) MiB = 120 MiB for K, V and the indexer at f16
constexpr uint32_t slack        = 512;  // mapped cells kept past the highest cell in use

uint32_t unit_from_env() {
    uint32_t unit = unit_default;
    if (const char * v = getenv("QNEXT_ELASTIC_UNIT"); v != nullptr && *v != '\0') {
        const long long u = atoll(v);
        if (u > 0) {
            unit = (uint32_t) std::min<long long>(u, 1 << 24);
        }
        unit = std::max<uint32_t>(256, (unit + 255) / 256 * 256);
    }
    return unit;
}

}

int llama_elastic_env_level() {
    static const int level = [] {
        const char * v = getenv("QNEXT_ELASTIC_KV");
        if (v == nullptr || *v == '\0') {
            return 0;
        }
        const int l = atoi(v);
        return std::clamp(l, 0, 3);
    }();
    return level;
}

llama_elastic_request_scope::llama_elastic_request_scope(int level) : prev(g_request) {
    g_request = level;
}

llama_elastic_request_scope::~llama_elastic_request_scope() {
    g_request = prev;
}

int llama_elastic_requested() {
    return g_request;
}

ggml_backend_buffer_type_t llama_elastic_buft_of(ggml_backend_buffer_type_t buft) {
    if (buft == nullptr) {
        return nullptr;
    }
    const llama_elastic_fns & api = llama_elastic_fns_of(ggml_backend_buft_get_device(buft));
    if (!api.resolved) {
        return nullptr;
    }
    return api.buft_of(buft);
}

llama_kv_elastic::llama_kv_elastic(const char * name, int level, uint32_t kv_size, std::vector<buffer_desc> descs) :
        name(name), level(level), kv_size(kv_size), unit(unit_from_env()), n_mapped(kv_size) {
    if (const char * v = getenv("QNEXT_MOE_STATS"); v != nullptr && atoll(v) > 0) {
        stats_every = atoll(v);
    }

    for (auto & d : descs) {
        const llama_elastic_fns & api = api_of_buf(d.buf);
        if (!api.resolved || !api.is_elastic(d.buf)) {
            throw std::runtime_error("qnext_elastic: the cache buffer is not elastic");
        }
        if (api.n_mapped(d.buf) != api.n_chunks(d.buf)) {
            throw std::runtime_error("qnext_elastic: the cache buffer is not fully mapped at load");
        }
        buf_state bs;
        bs.d = std::move(d);
        bufs.push_back(std::move(bs));
    }

    if (level >= 2 && kv_size > unit) {
        const auto t0 = std::chrono::steady_clock::now();

        try {
            for (auto & bs : bufs) {
                const llama_elastic_fns & api = api_of_buf(bs.d.buf);
                const size_t gran = api.granularity(bs.d.buf);
                const std::vector<uint8_t> keep = want_of(bs, unit);

                std::vector<size_t> src;
                for (size_t c = 0; c < keep.size(); ++c) {
                    if (!keep[c]) {
                        src.push_back(c);
                    }
                }
                if (src.empty()) {
                    continue;
                }

                bs.n_spare = src.size();
                bs.spare   = api.reserve(ggml_backend_buffer_get_type(bs.d.buf), bs.n_spare*gran);
                if (bs.spare == nullptr) {
                    throw std::runtime_error("qnext_elastic: failed to reserve the spare range");
                }
                std::vector<size_t> dst(src.size());
                for (size_t i = 0; i < dst.size(); ++i) {
                    dst[i] = i;
                }
                // the buffer was just cleared and synchronized at allocation, nothing reads it yet
                if (!api.move(bs.d.buf, src.data(), bs.spare, dst.data(), src.size(), false)) {
                    GGML_ABORT("qnext_elastic: %s: parking %zu chunks failed", this->name.c_str(), src.size());
                }
                bs.top = bs.n_spare;
            }
        } catch (...) {
            // the cache fails to construct: release what was parked so far (its buffer is unusable and freed next)
            for (auto & bs : bufs) {
                if (bs.spare) {
                    ggml_backend_buffer_free(bs.spare);
                    bs.spare = nullptr;
                }
            }
            throw;
        }
        n_mapped = unit;

        last_ms   = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        total_ms += last_ms;
        for (const auto & bs : bufs) {
            last_chunks += bs.n_spare;
        }
    }

    log("init");
}

llama_kv_elastic::~llama_kv_elastic() {
    log("final");
    for (auto & bs : bufs) {
        if (bs.spare) {
            ggml_backend_buffer_free(bs.spare);
            bs.spare = nullptr;
        }
    }
}

std::vector<uint8_t> llama_kv_elastic::want_of(const buf_state & bs, uint32_t cells) const {
    const llama_elastic_fns & api = api_of_buf(bs.d.buf);
    const size_t gran = api.granularity(bs.d.buf);
    const size_t n    = api.n_chunks(bs.d.buf);

    std::vector<uint8_t> want(n, 0);
    if (cells >= kv_size) {
        std::fill(want.begin(), want.end(), 1);
        return want;
    }
    for (const auto & t : bs.d.tensors) {
        const size_t end = t.offset + (size_t) cells*t.row;
        for (size_t c = t.offset/gran; c < n && c*gran < end; ++c) {
            want[c] = 1;
        }
    }
    return want;
}

void llama_kv_elastic::need(uint32_t n) {
    // keep at least `slack` mapped cells past the highest cell in use: the attention views n_kv = the used cells padded
    // to 256, so n_kv never reaches the end of the mapped cells
    if (n_mapped >= kv_size || (uint64_t) n + slack <= n_mapped) {
        return;
    }
    const uint64_t t = ((uint64_t) n + slack + unit - 1) / unit * unit + unit;
    map_to((uint32_t) std::min<uint64_t>(t, kv_size), "grow");
}

void llama_kv_elastic::need_all() {
    if (n_mapped < kv_size) {
        map_to(kv_size, "grow_all");
    }
}

void llama_kv_elastic::map_to(uint32_t cells, const char * why) {
    GGML_ASSERT(cells > n_mapped && cells <= kv_size);

    const auto t0 = std::chrono::steady_clock::now();
    size_t moved = 0;

    for (auto & bs : bufs) {
        const llama_elastic_fns & api = api_of_buf(bs.d.buf);
        const std::vector<uint8_t> want = want_of(bs, cells);

        std::vector<size_t> dst;
        for (size_t c = 0; c < want.size(); ++c) {
            if (want[c] && !api.chunk_mapped(bs.d.buf, c)) {
                dst.push_back(c);
            }
        }
        if (dst.empty()) {
            continue;
        }
        moved += dst.size();
        if (dst.size() > bs.top + bs.lent + bs.lost) {
            GGML_ABORT("qnext_elastic: %s: %zu chunks needed for %u cells, %zu parked + %zu lent + %zu lost: the chunk accounting is broken",
                name.c_str(), dst.size(), cells, bs.top, bs.lent, bs.lost);
        }

        // the parked chunks first: never read by the GPU, and the destination chunks are unmapped, so no
        // synchronization is needed; the zero fill is synchronous
        const size_t n_spare = std::min(dst.size(), bs.top);
        if (n_spare > 0) {
            std::vector<size_t> src(n_spare);
            for (size_t i = 0; i < n_spare; ++i) {
                src[i] = bs.top - 1 - i;
            }
            if (!api.move(bs.spare, src.data(), bs.d.buf, dst.data(), n_spare, true)) {
                GGML_ABORT("qnext_elastic: %s: mapping %zu chunks for %u cells failed (driver error)", name.c_str(), n_spare, cells);
            }
            bs.top -= n_spare;
        }

        // then the lent ones, from the bank tails (the lender synchronizes the device and parks the slots on them)
        size_t n_rest = dst.size() - n_spare;
        const size_t n_lent = lender ? std::min(n_rest, bs.lent) : 0;
        if (n_lent > 0) {
            lender->take(bs.d.buf, dst.data() + n_spare, n_lent);
            bs.lent -= n_lent;
            n_rest  -= n_lent;
        }

        // the lender went away (context teardown) with chunks of this buffer: their memory was released with its
        // banks, so these chunks get new memory (the VRAM is free again)
        if (n_rest > 0) {
            GGML_ASSERT(n_rest <= bs.lost);
            if (!api.map_new(bs.d.buf, dst.data() + n_spare + n_lent, n_rest)) {
                GGML_ABORT("qnext_elastic: %s: allocating %zu chunks after the lender went away failed", name.c_str(), n_rest);
            }
            bs.lost -= n_rest;
        }
    }

    if (cells >= kv_size) {
        for (const auto & bs : bufs) {
            GGML_ASSERT(bs.top == 0 && bs.lent == 0 && bs.lost == 0 && "qnext_elastic: chunks left outside with all cells mapped");
        }
    }

    n_mapped = cells;
    n_grow++;
    if (moved > 0) {
        last_ms     = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        last_chunks = moved;
        total_ms   += last_ms;
    }

    // a grow within chunks that are already mapped (a unit smaller than one chunk of some tensor) moves nothing
    log(moved > 0 ? why : "grow_in_place");
}

void llama_kv_elastic::maybe_shrink(uint32_t n) {
    if (lender == nullptr || level < 3) {
        return;
    }
    const uint64_t t = ((uint64_t) n + slack + unit - 1) / unit * unit + unit;
    if (t >= kv_size || (uint64_t) n_mapped < t + 2*(uint64_t) unit) {
        return;
    }
    shrink_to((uint32_t) t);
}

void llama_kv_elastic::shrink_to(uint32_t cells) {
    GGML_ASSERT(cells < n_mapped && cells >= unit);

    const auto t0 = std::chrono::steady_clock::now();
    size_t moved = 0;

    for (auto & bs : bufs) {
        const llama_elastic_fns & api = api_of_buf(bs.d.buf);
        const std::vector<uint8_t> want = want_of(bs, cells);

        std::vector<size_t> src;
        for (size_t c = 0; c < want.size(); ++c) {
            if (!want[c] && api.chunk_mapped(bs.d.buf, c)) {
                src.push_back(c);
            }
        }
        if (src.empty()) {
            continue;
        }

        moved += src.size();

        // the last graph may still read these cells (its n_kv was computed before the cells were freed)
        api.sync(bs.d.buf);

        const size_t n_given = lender->give(bs.d.buf, src.data(), src.size());
        bs.lent += n_given;

        // what the banks have no room for goes back to the spare range, which has room: the chunks outside the buffer
        // never exceed the ones parked at load
        const size_t n_rest = src.size() - n_given;
        if (n_rest > 0) {
            GGML_ASSERT(bs.spare != nullptr && bs.top + n_rest <= bs.n_spare);
            std::vector<size_t> dst(n_rest);
            for (size_t i = 0; i < n_rest; ++i) {
                dst[i] = bs.top + i;
            }
            if (!api.move(bs.d.buf, src.data() + n_given, bs.spare, dst.data(), n_rest, false)) {
                GGML_ABORT("qnext_elastic: %s: parking %zu chunks failed (driver error)", name.c_str(), n_rest);
            }
            bs.top += n_rest;
        }
    }

    n_mapped = cells;
    n_shrink++;
    last_ms     = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    last_chunks = moved;
    total_ms   += last_ms;

    log("shrink");
}

size_t llama_kv_elastic::parked_bytes() const {
    size_t res = 0;
    for (const auto & bs : bufs) {
        res += bs.top*api_of_buf(bs.d.buf).granularity(bs.d.buf);
    }
    return res;
}

void llama_kv_elastic::lend(llama_elastic_lender * l) {
    GGML_ASSERT(lender == nullptr && level >= 3);
    lender = l;
    lender->attach(this);

    const auto t0 = std::chrono::steady_clock::now();
    size_t moved = 0;
    for (auto & bs : bufs) {
        if (bs.top == 0) {
            continue;
        }
        // from the top of the stack, so that what the banks do not take stays a stack
        std::vector<size_t> src(bs.top);
        for (size_t i = 0; i < bs.top; ++i) {
            src[i] = bs.top - 1 - i;
        }
        const size_t n_given = lender->give(bs.spare, src.data(), src.size());
        bs.top  -= n_given;
        bs.lent += n_given;
        moved   += n_given;
    }
    last_ms     = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    last_chunks = moved;
    total_ms   += last_ms;

    log("lend");
}

void llama_kv_elastic::lender_gone() {
    // the lent chunks were released with the banks: no longer counted as lent, so no later grow asks the lender for them
    for (auto & bs : bufs) {
        bs.lost += bs.lent;
        bs.lent  = 0;
    }
    lender = nullptr;
}

void llama_kv_elastic::tick() {
    if (stats_every > 0 && ++ticks % stats_every == 0) {
        log("periodic");
    }
}

//
// QNEXT_ELASTIC_COMPUTE, see llama-elastic.h
//
// Invariant: while lent, the chunks [lent_first, buf_chunks) of lent_buf that are unmapped are exactly the n_lent chunks
// the MoE cache holds for this buffer (moved there by give, each zero filled at the destination); every other chunk of
// lent_buf is mapped. Every allocation of the scheduler is checked (check_alloc) before any tensor of it is written, so
// no tensor of a graph is ever placed in an unmapped chunk; the buffer is never freed while lent (the scheduler's buffer
// free callback takes back first), so the physical memory mapped never grows: only the same chunks move.

int llama_elastic_compute_env() {
    static const int v = [] {
        const char * s = getenv("QNEXT_ELASTIC_COMPUTE");
        return s != nullptr && strcmp(s, "1") == 0 ? 1 : 0;
    }();
    return v;
}

llama_compute_elastic::llama_compute_elastic(llama_elastic_lender * lender, const llama_elastic_fns & api, size_t gran, size_t slack) :
        lender(lender), api(api), gran(gran), slack(slack) {
    GGML_ASSERT(lender != nullptr && api.spans_ok() && gran > 0);
    // a KV growth/shrink that needs the bank tails while spans of the compute buffer sit on top takes them back first
    lender->set_span_evict(evict_cb, this);
}

llama_compute_elastic::~llama_compute_elastic() {
    // the compute buffer is freed with the scheduler and the banks with the MoE cache: whatever is lent at that point is
    // released with the banks (mappings are released by whoever has them mapped), nothing is taken back here
    lender->set_span_evict(nullptr, nullptr);
    if (n_give + n_take > 0 || off_reason != nullptr) {
        log("final");
    }
}

void llama_compute_elastic::evict_cb(void * ud) {
    auto * self = static_cast<llama_compute_elastic *>(ud);
    if (self->n_lent > 0) {
        self->n_evict++;
        self->take(self->lent_buf, "take (the KV cache grows or shrinks: the bank tails are needed)");
    }
}

void llama_compute_elastic::disable(const char * why) {
    if (off_reason == nullptr) {
        off_reason = why;
        LLAMA_LOG_WARN("qnext_elastic_compute: lending of the compute buffer disabled: %s\n", why);
    }
}

void llama_compute_elastic::set_keep(size_t high_water, const char * what) {
    // a measurement replaces the keep region (it may shrink, e.g. when backend samplers go away), never below the floor
    // the safety net set; the lent spans start at lent_first, fixed at the give: a smaller keep region leaves them as
    // they are until the next give, a larger one that reaches them (+ the guard) needs them back first
    const size_t k = std::max(keep_floor, (high_water + slack + gran - 1)/gran*gran);
    if (k == keep_bytes) {
        return;
    }
    if (n_lent > 0 && k + guard_chunks*gran > lent_first*gran) {
        take(lent_buf, "take (keep region grew)");
    }
    LLAMA_LOG_INFO("qnext_elastic_compute: keep region %.1f -> %.1f MiB (high water %.2f MiB + slack %.1f MiB, floor %.1f MiB), "
        "set by: %s\n", keep_bytes/1024.0/1024.0, k/1024.0/1024.0, high_water/1024.0/1024.0, slack/1024.0/1024.0,
        keep_floor/1024.0/1024.0, what);
    keep_bytes = k;
}

size_t llama_compute_elastic::give(ggml_backend_buffer_t buf) {
    if (disabled() || n_lent > 0 || buf == nullptr || keep_bytes == 0) {
        return 0;
    }
    if (!api.is_elastic(buf)) {
        disable("the compute buffer is not an elastic buffer");
        return 0;
    }
    const size_t n    = api.n_chunks(buf);
    const size_t need = keep_bytes/gran + guard_chunks; // chunks [0, need) stay mapped
    // the mappings (spans) of the buffer whose head is at or above `need` (a span across the boundary stays), from
    // the top down: if the banks take only part, what stays mapped is contiguous with the keep region
    std::vector<size_t> heads;
    std::vector<size_t> lens;
    for (size_t c = 0; c < n; ) {
        const size_t k = api.span_of(buf, c);
        if (k == 0) {
            GGML_ABORT("qnext_elastic_compute: chunk %zu of the compute buffer is not the head of a mapping while nothing is lent", c);
        }
        if (c >= need) {
            heads.push_back(c);
            lens.push_back(k);
        }
        c += k;
    }
    if (heads.empty()) {
        return 0;
    }
    std::reverse(heads.begin(), heads.end());
    std::reverse(lens.begin(), lens.end());

    const auto t0 = std::chrono::steady_clock::now();
    // the last graphs may still read or write the region (and the MoE cache upload streams the banks)
    api.sync(buf);
    const size_t  l0 = lender->lent_chunks();
    const int64_t s0 = lender->lent_slots();
    size_t calls = 0;
    const size_t got = lender->give_spans(buf, heads.data(), heads.size(), &calls);
    last_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (got == 0) {
        disable("the MoE cache banks have no room for lent spans (QNEXT_ELASTIC_COMPUTE_VA_MIB)");
        return 0;
    }
    size_t chunks = 0;
    for (size_t i = 0; i < got; ++i) {
        chunks += lens[i];
    }
    GGML_ASSERT(lender->lent_chunks() == l0 + chunks && "qnext_elastic_compute: the MoE cache chunk accounting is broken");

    lent_heads.assign(heads.begin(), heads.begin() + got);
    lent_lens.assign(lens.begin(), lens.begin() + got);
    lent_buf   = buf;
    lent_first = lent_heads.back(); // the lowest: the list is from the top down
    buf_chunks = n;
    n_lent     = chunks;
    last_slots = lender->lent_slots() - s0;
    last_calls = calls;
    n_give++;
    ms_give += last_ms;

    if (n_give <= 4 || n_give % 32 == 0) {
        log(got < heads.size() ? "give (partial: the banks are full)" : "give");
    }
    return chunks;
}

void llama_compute_elastic::take(ggml_backend_buffer_t buf, const char * why) {
    if (n_lent == 0) {
        return;
    }
    if (buf != lent_buf || api.n_chunks(buf) != buf_chunks) {
        GGML_ABORT("qnext_elastic_compute: taking back into a buffer the chunks were not lent from (%s)", why);
    }
    for (size_t i = 0; i < lent_heads.size(); ++i) {
        for (size_t j = 0; j < lent_lens[i]; ++j) {
            if (api.chunk_mapped(buf, lent_heads[i] + j)) {
                GGML_ABORT("qnext_elastic_compute: lent chunk %zu is mapped: the chunk accounting is broken", lent_heads[i] + j);
            }
        }
    }
    if (lender->n_spans() != lent_heads.size() || lender->lent_chunks() < n_lent) {
        GGML_ABORT("qnext_elastic_compute: %zu spans lent, the banks hold %zu spans and %zu chunks: the accounting is broken",
            lent_heads.size(), lender->n_spans(), lender->lent_chunks());
    }

    const auto t0 = std::chrono::steady_clock::now();
    const uint64_t d0 = lender->dropped();
    // synchronizes the device, parks the bank slots on the spans (their experts are dropped), moves them back zero filled
    size_t calls = 0;
    lender->take_spans(buf, lent_heads.data(), lent_lens.data(), lent_heads.size(), &calls);
    last_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    last_dropped   = lender->dropped() - d0;
    dropped_total += last_dropped;
    last_calls     = calls;
    n_lent   = 0;
    lent_buf = nullptr;
    lent_heads.clear();
    lent_lens.clear();
    n_take++;
    ms_take += last_ms;

    if (n_take <= 4 || n_take % 32 == 0) {
        log(why);
    }
}

bool llama_compute_elastic::check_alloc(ggml_backend_buffer_t buf, size_t high_water, int64_t n_tokens, int64_t max_batch) {
    if (n_lent == 0) {
        return false;
    }
    if (buf != lent_buf) {
        GGML_ABORT("qnext_elastic_compute: the compute buffer changed while lent");
    }
    if (high_water != SIZE_MAX && high_water <= keep_bytes) {
        return false;
    }
    const bool small = n_tokens >= 1 && n_tokens <= max_batch;
    if (!small) {
        take(buf, n_tokens > max_batch ? "take (large graph)" : "take (other graph: K-shift, reserve or unknown size)");
        return true;
    }
    // the safety net: a small graph whose assignment reaches past the keep region
    n_net++;
    if (!warned_net) {
        warned_net = true;
        if (high_water == SIZE_MAX) {
            LLAMA_LOG_WARN("qnext_elastic_compute: a %lld-token graph uses more than one compute buffer chunk: taking the lent "
                "memory back\n", (long long) n_tokens);
        } else {
            LLAMA_LOG_WARN("qnext_elastic_compute: a %lld-token graph needs %.2f MiB of the compute buffer, past the keep "
                "region of %.2f MiB: taking the lent memory back and raising the keep region (once per process)\n",
                (long long) n_tokens, high_water/1024.0/1024.0, keep_bytes/1024.0/1024.0);
        }
    }
    take(buf, "take (safety net: small graph past the keep region)");
    if (high_water == SIZE_MAX) {
        disable("a small graph uses more than one compute buffer chunk");
    } else {
        keep_floor = std::max(keep_floor, (high_water + slack + gran - 1)/gran*gran);
        char what[128];
        snprintf(what, sizeof(what), "safety net, a %lld-token graph (kept as the floor)", (long long) n_tokens);
        set_keep(high_water, what);
        log("keep raised");
    }
    return true;
}

void llama_compute_elastic::log(const char * event) const {
    constexpr double MiB = 1024.0*1024.0;
    LLAMA_LOG_INFO("qnext_elastic_compute: %s: keep=%.1f MiB guard=%.1f MiB lent=%.1f MiB (%zu chunks) slots_gained_last_give=%lld "
        "gives=%llu takes=%llu net=%llu kv_evicts=%llu last_ms=%.2f last_calls=%llu spans=%zu span=%.0f MiB give_ms_avg=%.2f "
        "take_ms_avg=%.2f dropped_last_take=%llu dropped_total=%llu%s%s cache=%s\n",
        event, keep_bytes/MiB, guard_chunks*gran/MiB, n_lent*gran/MiB, n_lent, (long long) last_slots,
        (unsigned long long) n_give, (unsigned long long) n_take, (unsigned long long) n_net, (unsigned long long) n_evict,
        last_ms, (unsigned long long) last_calls, lent_heads.size(), lent_lens.empty() ? 0.0 : lent_lens[0]*gran/MiB,
        n_give ? ms_give/n_give : 0.0, n_take ? ms_take/n_take : 0.0,
        (unsigned long long) last_dropped, (unsigned long long) dropped_total,
        off_reason ? " disabled: " : "", off_reason ? off_reason : "", lender->describe().c_str());
}

uint32_t llama_kv_elastic::cells_backed() const {
    uint64_t res = kv_size;
    for (const auto & bs : bufs) {
        const llama_elastic_fns & api = api_of_buf(bs.d.buf);
        const size_t gran = api.granularity(bs.d.buf);
        const size_t n    = api.n_chunks(bs.d.buf);
        for (const auto & t : bs.d.tensors) {
            size_t c = t.offset/gran;
            while (c < n && api.chunk_mapped(bs.d.buf, c)) {
                ++c;
            }
            const size_t end = std::min(c*gran, t.offset + t.size);
            res = std::min<uint64_t>(res, end > t.offset ? (end - t.offset)/t.row : 0);
        }
    }
    return (uint32_t) res;
}

// cells = the cells the policy guarantees mapped; backed = the cells actually covered by mapped chunks in every tensor
// (more than cells when the unit is not a whole number of chunks); kv_mapped = the chunks mapped in the cache buffers,
// which includes chunks shared by neighbouring tensors; event_ms/event_chunks = the last event that moved chunks
void llama_kv_elastic::log(const char * event) const {
    size_t mapped = 0;
    size_t total  = 0;
    size_t spare  = 0;
    size_t lent   = 0;
    size_t lost   = 0;
    for (const auto & bs : bufs) {
        const llama_elastic_fns & api = api_of_buf(bs.d.buf);
        const size_t gran = api.granularity(bs.d.buf);
        mapped += api.n_mapped(bs.d.buf)*gran;
        total  += api.n_chunks(bs.d.buf)*gran;
        if (bs.spare) {
            spare += api.n_mapped(bs.spare)*gran;
        }
        lent += bs.lent*gran;
        lost += bs.lost*gran;
    }
    constexpr double MiB = 1024.0*1024.0;
    const std::string cache = lender ? lender->describe() : std::string("slots=0+0 dropped=0");
    LLAMA_LOG_INFO("qnext_elastic: %s cache=%s level=%d cells=%u/%u backed=%u unit=%u kv_mapped=%.1f/%.1f MiB spare=%.1f MiB "
        "lent=%.1f MiB%s grows=%llu shrinks=%llu event_ms=%.3f event_chunks=%zu total_ms=%.3f %s relocated=0\n",
        event, name.c_str(), level, n_mapped, kv_size, cells_backed(), unit, mapped/MiB, total/MiB, spare/MiB,
        lent/MiB, lost > 0 ? " (lender gone)" : "",
        (unsigned long long) n_grow, (unsigned long long) n_shrink, last_ms, last_chunks, total_ms, cache.c_str());
}
