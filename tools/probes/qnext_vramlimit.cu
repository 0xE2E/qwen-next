// qnext_vramlimit: how much VRAM can a CUDA process on this WDDM machine use at full speed?
// Phase 1: allocate --start-mib in 1 GiB pieces (cudaMalloc, or VMM: cuMemCreate+cuMemMap+cuMemSetAccess into one
//          reserved range), write every piece, measure reference read bandwidth.
// Phase 2: add --step-mib chunks; per chunk measure alloc time, write GB/s, read GB/s, early-piece read GB/s, cudaMemGetInfo.
// Stops at: allocation error | slowdown (< ref/slow-factor) | --max-mib. Frees everything on every exit path.
// Usage: qnext_vramlimit [--start-mib 13000] [--step-mib 64] [--max-mib 16150] [--mode malloc|vmm]
//                        [--slow-factor 3] [--hold-ms 0] [--out file.json]
// stdout: line protocol (START / REF / STEP / STOP / END), flushed per line. Not a benchmark of the engine.
#include <cuda_runtime.h>
#include <cuda.h>
#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using clk = std::chrono::steady_clock;
static double now_ms() { return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count(); }
static std::string fmt(const char* f, ...) {
    char b[4096]; va_list a; va_start(a, f); vsnprintf(b, sizeof b, f, a); va_end(a); return b;
}
static const size_t MiB = 1ull << 20, GiB = 1ull << 30;
static volatile sig_atomic_t g_stop = 0;
static void on_sigint(int) { g_stop = 1; }

// ---------- kernels ----------
__device__ __forceinline__ unsigned pat(size_t i, unsigned seed) { return (unsigned)i * 2654435761u ^ seed; }
__global__ void k_write(uint4* p, size_t n, unsigned seed) {  // n in uint4; word index = 4*i+k
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        p[i] = make_uint4(pat(4 * i, seed), pat(4 * i + 1, seed), pat(4 * i + 2, seed), pat(4 * i + 3, seed));
}
__global__ void k_sum(const uint4* p, size_t n, unsigned long long* out) {
    unsigned s = 0;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x, st = (size_t)gridDim.x * blockDim.x;
#pragma unroll 4
    for (; i < n; i += st) { uint4 x = p[i]; s += x.x ^ x.y ^ x.z ^ x.w; }
    if (s) atomicAdd(out, (unsigned long long)s);  // keep the loads alive
}
__global__ void k_check(const uint4* p, size_t n, unsigned seed, unsigned long long* mism) {
    unsigned long long m = 0;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint4 x = p[i];
        m += (x.x != pat(4 * i, seed)) + (x.y != pat(4 * i + 1, seed)) + (x.z != pat(4 * i + 2, seed)) + (x.w != pat(4 * i + 3, seed));
    }
    if (m) atomicAdd(mism, m);
}

// ---------- state ----------
struct Block { char* ptr = nullptr; size_t bytes = 0; CUmemGenericAllocationHandle h = 0; bool vmm = false; bool mapped = false; };
static std::vector<Block> g_blocks;     // all live blocks in allocation order
static bool g_vmm = false;
static CUdeviceptr g_base = 0; static size_t g_range = 0, g_off = 0, g_gran = 2 * MiB;
static unsigned long long* g_out = nullptr;
static int g_blocksPerGrid = 0;

static size_t round_up(size_t v, size_t a) { return (v + a - 1) / a * a; }
static std::string cu_name(CUresult r) { const char* s = "?"; cuGetErrorName(r, &s); return s ? s : "?"; }

// returns 0 on success; else error code, name in `name`
static int alloc_block(size_t bytes, Block& b, std::string& name) {
    if (!g_vmm) {
        void* p = nullptr; cudaError_t e = cudaMalloc(&p, bytes);
        if (e != cudaSuccess) { cudaGetLastError(); name = cudaGetErrorName(e); return (int)e; }
        b.ptr = (char*)p; b.bytes = bytes; name = "cudaSuccess"; return 0;
    }
    bytes = round_up(bytes, g_gran);
    if (g_off + bytes > g_range) { name = "vmm_range_exhausted"; return -1; }
    CUmemAllocationProp pr = {}; pr.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    pr.location.type = CU_MEM_LOCATION_TYPE_DEVICE; pr.location.id = 0;
    CUmemGenericAllocationHandle h = 0;
    CUresult r = cuMemCreate(&h, bytes, &pr, 0);
    if (r != CUDA_SUCCESS) { name = "cuMemCreate:" + cu_name(r); return (int)r; }
    r = cuMemMap(g_base + g_off, bytes, 0, h, 0);
    if (r != CUDA_SUCCESS) { cuMemRelease(h); name = "cuMemMap:" + cu_name(r); return (int)r; }
    CUmemAccessDesc d = {}; d.location.type = CU_MEM_LOCATION_TYPE_DEVICE; d.location.id = 0; d.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    r = cuMemSetAccess(g_base + g_off, bytes, &d, 1);
    if (r != CUDA_SUCCESS) { cuMemUnmap(g_base + g_off, bytes); cuMemRelease(h); name = "cuMemSetAccess:" + cu_name(r); return (int)r; }
    b.ptr = (char*)(uintptr_t)(g_base + g_off); b.bytes = bytes; b.h = h; b.vmm = true; b.mapped = true;
    g_off += bytes; name = "CUDA_SUCCESS"; return 0;
}

static void free_all() {
    cudaDeviceSynchronize(); cudaGetLastError();
    for (size_t i = g_blocks.size(); i-- > 0;) {
        Block& b = g_blocks[i];
        if (b.vmm) { if (b.mapped) cuMemUnmap((CUdeviceptr)(uintptr_t)b.ptr, b.bytes); if (b.h) cuMemRelease(b.h); }
        else if (b.ptr) cudaFree(b.ptr);
    }
    g_blocks.clear();
    if (g_base) { cuMemAddressFree(g_base, g_range); g_base = 0; }
    if (g_out) { cudaFree(g_out); g_out = nullptr; }
}

// ---------- measurements ----------
static bool g_err = false; static std::string g_errname;
static bool cuda_ok(const char* where) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { g_err = true; g_errname = std::string(where) + ":" + cudaGetErrorName(e); return false; }
    return true;
}
static double med(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
static cudaEvent_t g_e0, g_e1;
// returns GB/s (1e9 B/s) or -1 on error
static double timed_read(const char* p, size_t bytes) {
    cudaEventRecord(g_e0);
    k_sum<<<g_blocksPerGrid, 256>>>((const uint4*)p, bytes / 16, g_out);
    cudaEventRecord(g_e1);
    if (cudaEventSynchronize(g_e1) != cudaSuccess || !cuda_ok("read")) return -1;
    float ms = 0; cudaEventElapsedTime(&ms, g_e0, g_e1);
    return bytes / (ms * 1e6);
}
static double timed_write(char* p, size_t bytes, unsigned seed) {
    cudaEventRecord(g_e0);
    k_write<<<g_blocksPerGrid, 256>>>((uint4*)p, bytes / 16, seed);
    cudaEventRecord(g_e1);
    if (cudaEventSynchronize(g_e1) != cudaSuccess || !cuda_ok("write")) return -1;
    float ms = 0; cudaEventElapsedTime(&ms, g_e0, g_e1);
    return bytes / (ms * 1e6);
}
static unsigned long long check_block(const Block& b, unsigned seed) {
    unsigned long long m = ~0ull;
    cudaMemset(g_out, 0, 8);
    k_check<<<g_blocksPerGrid, 256>>>((const uint4*)b.ptr, b.bytes / 16, seed, g_out);
    if (cudaDeviceSynchronize() == cudaSuccess && cuda_ok("check")) cudaMemcpy(&m, g_out, 8, cudaMemcpyDeviceToHost);
    return m;
}
static void free_info(size_t& fr, size_t& tot) { fr = tot = 0; cudaMemGetInfo(&fr, &tot); }

struct Step { double cum_mib, alloc_ms, write_gbps, read_gbps, early_gbps; double free_mib; int rc; std::string rcname; };

int main(int argc, char** argv) {
    size_t start_mib = 13000, step_mib = 64, max_mib = 16150; int hold_ms = 0; double slow = 3.0;
    std::string mode = "malloc", outp;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i]; auto nx = [&]() -> const char* { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(2); } return argv[++i]; };
        if (a == "--start-mib") start_mib = strtoull(nx(), 0, 10);
        else if (a == "--step-mib") step_mib = strtoull(nx(), 0, 10);
        else if (a == "--max-mib") max_mib = strtoull(nx(), 0, 10);
        else if (a == "--mode") mode = nx();
        else if (a == "--slow-factor") slow = atof(nx());
        else if (a == "--hold-ms") hold_ms = atoi(nx());
        else if (a == "--out") outp = nx();
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    if (mode != "malloc" && mode != "vmm") { fprintf(stderr, "bad --mode\n"); return 2; }
    if (step_mib == 0 || start_mib == 0) { fprintf(stderr, "bad sizes\n"); return 2; }
    g_vmm = mode == "vmm";
    signal(SIGINT, on_sigint);

    if (cudaSetDevice(0) != cudaSuccess || cudaFree(0) != cudaSuccess) { fprintf(stderr, "cuda init failed\n"); return 1; }
    cuInit(0);
    int sms = 0; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0); g_blocksPerGrid = sms * 8;
    size_t f0, tot; free_info(f0, tot);
    printf("START mode=%s free_mib=%.1f total_mib=%.1f start_mib=%zu step_mib=%zu max_mib=%zu slow=%.2f\n", mode.c_str(), f0 / (double)MiB, tot / (double)MiB, start_mib, step_mib, max_mib, slow);
    fflush(stdout);

    std::vector<Step> steps; std::string stop = "none"; std::string stop_detail;
    double ref_read = 0, ref_write = 0, last_full = 0, cum_mib = 0; double thr = 0;
    unsigned long long mism_last = 0, mism_early = 0; bool integ_done = false;
    std::vector<size_t> piece_bytes; std::vector<unsigned> seeds;

    cudaEventCreate(&g_e0); cudaEventCreate(&g_e1);
    if (cudaMalloc((void**)&g_out, 64) != cudaSuccess) { fprintf(stderr, "scratch alloc failed\n"); return 1; }
    cudaMemset(g_out, 0, 64);

    if (g_vmm) {
        CUmemAllocationProp pr = {}; pr.type = CU_MEM_ALLOCATION_TYPE_PINNED; pr.location.type = CU_MEM_LOCATION_TYPE_DEVICE; pr.location.id = 0;
        size_t gm = 0; if (cuMemGetAllocationGranularity(&gm, &pr, CU_MEM_ALLOC_GRANULARITY_MINIMUM) == CUDA_SUCCESS && gm) g_gran = gm;
        g_range = round_up((max_mib + step_mib + 1024) * MiB, g_gran);   // address space only, not memory
        CUresult r = cuMemAddressReserve(&g_base, g_range, 0, 0, 0);
        if (r != CUDA_SUCCESS) { printf("STOP reason=vmm_reserve_failed detail=%s\n", cu_name(r).c_str()); fflush(stdout); free_all(); return 1; }
        printf("INFO vmm_granularity=%zu reserved_mib=%zu\n", g_gran, g_range / MiB); fflush(stdout);
    }

    // ---- Phase 1 ----
    {
        size_t remaining = start_mib * MiB; unsigned idx = 0;
        while (remaining && !g_stop) {
            size_t sz = std::min(remaining, GiB); sz = round_up(sz, 2 * MiB);
            Block b; std::string nm; int rc = alloc_block(sz, b, nm);
            if (rc) { stop = "phase1_alloc_error"; stop_detail = nm; break; }
            g_blocks.push_back(b); piece_bytes.push_back(b.bytes); seeds.push_back(idx);
            if (timed_write(b.ptr, b.bytes, idx) < 0) { stop = "phase1_cuda_error"; stop_detail = g_errname; break; }
            cum_mib += b.bytes / (double)MiB; remaining = remaining > sz ? remaining - sz : 0; idx++;
        }
        if (stop == "none" && !g_stop && !g_blocks.empty()) {
            size_t rb = std::min<size_t>(512 * MiB, g_blocks[0].bytes);
            std::vector<double> rr, rw;
            for (int i = 0; i < 5 && !g_err; i++) { double g = timed_read(g_blocks[0].ptr, rb); if (g < 0) break; rr.push_back(g); }
            for (int i = 0; i < 3 && !g_err; i++) { double g = timed_write(g_blocks[0].ptr, rb, seeds[0]); if (g < 0) break; rw.push_back(g); }  // same seed: keeps piece 0 pattern
            if (g_err || rr.empty()) { stop = "phase1_cuda_error"; stop_detail = g_errname; }
            else { ref_read = med(rr); ref_write = rw.empty() ? 0 : med(rw); thr = ref_read / slow; }
        }
        last_full = cum_mib;
        double fr, t_; { size_t a, b; free_info(a, b); fr = a / (double)MiB; t_ = b / (double)MiB; }
        printf("REF cum_mib=%.1f read_gbps=%.1f write_gbps=%.1f threshold_gbps=%.1f cuda_free_mib=%.1f cuda_total_mib=%.1f\n", cum_mib, ref_read, ref_write, thr, fr, t_);
        fflush(stdout);
        if (g_stop && stop == "none") stop = "interrupted";
    }

    // ---- Phase 2 ----
    unsigned cidx = 0;
    while (stop == "none") {
        if (g_stop) { stop = "interrupted"; break; }
        if (cum_mib >= (double)max_mib) { stop = "max_mib_reached"; break; }
        size_t csz = std::min<size_t>(step_mib, (size_t)(max_mib - cum_mib > 0 ? max_mib - cum_mib : step_mib)) * MiB;
        if (csz < 2 * MiB) csz = 2 * MiB;
        csz = round_up(csz, 2 * MiB);
        Block b; std::string nm; double t0 = now_ms();
        int rc = alloc_block(csz, b, nm); double alloc_ms = now_ms() - t0;
        Step s{}; s.rc = rc; s.rcname = nm; s.alloc_ms = alloc_ms;
        if (rc) {
            size_t a, t2; free_info(a, t2); s.cum_mib = cum_mib; s.free_mib = a / (double)MiB; s.write_gbps = s.read_gbps = s.early_gbps = 0;
            steps.push_back(s);
            printf("STEP cum_mib=%.0f alloc_ms=%.3f write_gbps=0 read_gbps=0 early_gbps=0 cuda_free_mib=%.1f rc=%s(%d)\n", cum_mib, alloc_ms, s.free_mib, nm.c_str(), rc);
            fflush(stdout);
            stop = "alloc_error"; stop_detail = nm; break;
        }
        g_blocks.push_back(b); cum_mib += b.bytes / (double)MiB; unsigned seed = 1000 + cidx++;
        double wg = timed_write(b.ptr, b.bytes, seed);
        double rg = -1, eg = -1;
        if (wg >= 0) {
            std::vector<double> v;
            size_t flush = std::min<size_t>(256 * MiB, g_blocks[0].bytes);
            for (int i = 0; i < 3; i++) { timed_read(g_blocks[0].ptr, flush); double g = timed_read(b.ptr, b.bytes); if (g < 0) break; v.push_back(g); }  // flush L2 first
            if (v.size() == 3) rg = med(v);
            if (rg >= 0) {
                std::vector<double> e; size_t rb = std::min<size_t>(512 * MiB, g_blocks[0].bytes);
                for (int i = 0; i < 3; i++) { double g = timed_read(g_blocks[0].ptr, rb); if (g < 0) break; e.push_back(g); }
                if (e.size() == 3) eg = med(e);
            }
        }
        size_t a, t2; free_info(a, t2);
        s.cum_mib = cum_mib; s.write_gbps = wg < 0 ? 0 : wg; s.read_gbps = rg < 0 ? 0 : rg; s.early_gbps = eg < 0 ? 0 : eg; s.free_mib = a / (double)MiB;
        steps.push_back(s);
        printf("STEP cum_mib=%.0f alloc_ms=%.3f write_gbps=%.1f read_gbps=%.1f early_gbps=%.1f cuda_free_mib=%.1f rc=%s(%d)\n", cum_mib, alloc_ms, s.write_gbps, s.read_gbps, s.early_gbps, s.free_mib, nm.c_str(), rc);
        fflush(stdout);
        if (g_err || wg < 0 || rg < 0 || eg < 0) { stop = "cuda_error"; stop_detail = g_errname; break; }
        if (wg < thr) { stop = "new_chunk_write_slow"; stop_detail = fmt("%.1f<%.1f", wg, thr); break; }
        if (rg < thr) { stop = "new_chunk_read_slow"; stop_detail = fmt("%.1f<%.1f", rg, thr); break; }
        if (eg < thr) { stop = "early_piece_read_slow"; stop_detail = fmt("%.1f<%.1f", eg, thr); break; }
        last_full = cum_mib;
        if (hold_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(hold_ms));
    }
    printf("STOP reason=%s detail=%s last_full_speed_mib=%.0f\n", stop.c_str(), stop_detail.c_str(), last_full); fflush(stdout);

    // ---- integrity: last allocated block and oldest piece ----
    if (!g_err && g_blocks.size() > 1 && g_out) {
        size_t np = piece_bytes.size();
        if (g_blocks.size() > np) { mism_last = check_block(g_blocks.back(), 1000 + cidx - 1); }
        else mism_last = check_block(g_blocks.back(), seeds.back());
        mism_early = check_block(g_blocks[0], seeds[0]);
        integ_done = true;
        printf("INTEGRITY last_block_mismatch_words=%llu piece0_mismatch_words=%llu\n", mism_last, mism_early); fflush(stdout);
    }

    size_t f1, t1; double fe_mib;
    free_all(); cudaDeviceSynchronize(); free_info(f1, t1); fe_mib = f1 / (double)MiB;
    printf("END free_mib=%.1f leak_mib=%.1f\n", fe_mib, (double)((long long)f0 - (long long)f1) / MiB); fflush(stdout);

    if (!outp.empty()) {
        std::string j = fmt("{\"mode\":\"%s\",\"start_mib\":%zu,\"step_mib\":%zu,\"max_mib\":%zu,\"slow_factor\":%.2f,\"cuda_total_mib\":%.1f,"
                            "\"start_free_mib\":%.1f,\"end_free_mib\":%.1f,\"leak_mib\":%.1f,\"ref_read_gbps\":%.2f,\"ref_write_gbps\":%.2f,"
                            "\"threshold_gbps\":%.2f,\"stop_reason\":\"%s\",\"stop_detail\":\"%s\",\"last_full_speed_mib\":%.0f,",
                            mode.c_str(), start_mib, step_mib, max_mib, slow, tot / (double)MiB, f0 / (double)MiB, fe_mib,
                            (double)((long long)f0 - (long long)f1) / MiB, ref_read, ref_write, thr, stop.c_str(), stop_detail.c_str(), last_full);
        if (integ_done) j += fmt("\"integrity\":{\"last_block_mismatch_words\":%llu,\"piece0_mismatch_words\":%llu},", mism_last, mism_early);
        j += "\"steps\":[";
        for (size_t i = 0; i < steps.size(); i++) {
            const Step& s = steps[i];
            j += fmt("%s{\"cum_mib\":%.0f,\"alloc_ms\":%.3f,\"write_gbps\":%.2f,\"read_gbps\":%.2f,\"early_read_gbps\":%.2f,\"cuda_free_mib\":%.1f,\"rc\":%d,\"rc_name\":\"%s\"}",
                     i ? "," : "", s.cum_mib, s.alloc_ms, s.write_gbps, s.read_gbps, s.early_gbps, s.free_mib, s.rc, s.rcname.c_str());
        }
        j += "]}\n";
        FILE* f = fopen(outp.c_str(), "wb");
        if (f) { fwrite(j.data(), 1, j.size(), f); fclose(f); } else fprintf(stderr, "cannot write %s\n", outp.c_str());
    }
    cudaEventDestroy(g_e0); cudaEventDestroy(g_e1);
    cudaDeviceReset();
    return 0;
}
