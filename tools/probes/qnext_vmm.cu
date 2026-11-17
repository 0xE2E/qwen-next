// qnext_vmm: standalone CUDA virtual-memory-management (VMM) viability probe.
// JSON on stdout, progress on stderr. Every section/sub-section is independent and records
// {"worked":false,"error":"..."} on failure. Peak GPU use ~3.5 GiB, pinned host 1 GiB.
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdarg>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using clk = std::chrono::steady_clock;
static double now_us() { return std::chrono::duration<double, std::micro>(clk::now().time_since_epoch()).count(); }
static std::string fmt(const char* f, ...) {
    char b[2048]; va_list a; va_start(a, f); vsnprintf(b, sizeof b, f, a); va_end(a); return b;
}
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) \
    throw std::runtime_error(fmt("%s failed: %s (%s:%d)", #x, cudaGetErrorString(e_), __FILE__, __LINE__)); } while (0)
#define CKD(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char* s_ = "?"; cuGetErrorString(r_, &s_); \
    throw std::runtime_error(fmt("%s failed: %s (%d) (line %d)", #x, s_, (int)r_, __LINE__)); } } while (0)

static std::string st(std::vector<double> v) {
    if (v.empty()) return "null";
    std::sort(v.begin(), v.end());
    return fmt("{\"min\":%.3f,\"median\":%.3f,\"p90\":%.3f,\"n\":%zu}", v.front(), v[v.size() / 2],
               v[std::min(v.size() - 1, (size_t)(v.size() * 0.9))], v.size());
}
static double med(std::vector<double> v) { if (v.empty()) return 0; std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
static double mn(const std::vector<double>& v) { return v.empty() ? 0 : *std::min_element(v.begin(), v.end()); }
static std::string jerr(const std::string& m) {
    std::string e; for (char c : m) { if (c == '"' || c == '\\') e += '\\'; if (c == '\n') c = ' '; e += c; }
    return "{\"worked\":false,\"error\":\"" + e + "\"}";
}
// run f (returns a JSON object string), append "key":obj to j; on exception record worked:false
static void add(std::string& j, const char* key, const std::function<std::string()>& f) {
    std::string r;
    fprintf(stderr, "  [%s]\n", key);
    try { r = f(); } catch (std::exception& x) { cudaGetLastError(); r = jerr(x.what()); }
    if (j.size() > 1) j += ",";
    j += std::string("\"") + key + "\":" + r;
}

static const size_t MiB = 1u << 20, GiB = 1ull << 30;
static const size_t CHUNK = 2 * MiB;
static const int NH = 1320;
static const size_t SLOT = 2034240;          // ~1.94 MiB, multiple of 16
static const int NSLOT = 141;
static std::vector<size_t> g_offs;           // 141 random distinct slot offsets inside 1 GiB
static size_t g_start_free = 0, g_total = 0;

// ---------- RAII ----------
struct Stream { cudaStream_t s = nullptr; Stream() { CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)); }
    ~Stream() { if (s) { cudaStreamSynchronize(s); cudaStreamDestroy(s); } } operator cudaStream_t() const { return s; } };
struct DevBuf { void* p = nullptr; explicit DevBuf(size_t n) { CK(cudaMalloc(&p, n)); } ~DevBuf() { cudaFree(p); } };
struct PinBuf { void* p = nullptr; explicit PinBuf(size_t n, unsigned fl = cudaHostAllocDefault) { CK(cudaHostAlloc(&p, n, fl)); } ~PinBuf() { if (p) cudaFreeHost(p); } };
struct Event { cudaEvent_t e = nullptr; Event() { CK(cudaEventCreate(&e)); } ~Event() { cudaEventDestroy(e); } };
struct Graph { cudaGraph_t g = nullptr; cudaGraphExec_t x = nullptr;
    ~Graph() { if (x) cudaGraphExecDestroy(x); if (g) cudaGraphDestroy(g); } };
static double ev_ms(Event& a, Event& b) { float ms = 0; CK(cudaEventElapsedTime(&ms, a.e, b.e)); return ms; }

// ---------- VMM helpers ----------
static CUmemAllocationProp mkprop() {
    CUmemAllocationProp p = {}; p.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    p.location.type = CU_MEM_LOCATION_TYPE_DEVICE; p.location.id = 0; return p;
}
static CUmemAccessDesc mkacc() {
    CUmemAccessDesc d = {}; d.location.type = CU_MEM_LOCATION_TYPE_DEVICE; d.location.id = 0;
    d.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE; return d;
}
struct Handles {
    std::vector<CUmemGenericAllocationHandle> h;
    Handles() = default; Handles(const Handles&) = delete; Handles& operator=(const Handles&) = delete;
    ~Handles() { for (auto x : h) if (x) cuMemRelease(x); }
    void create(size_t n, size_t sz) {
        CUmemAllocationProp p = mkprop();
        for (size_t i = 0; i < n; i++) { CUmemGenericAllocationHandle x = 0; CKD(cuMemCreate(&x, sz, &p, 0)); h.push_back(x); }
    }
};
struct Range {
    CUdeviceptr base = 0; size_t chunk, n; std::vector<char> mp;
    Range(size_t n_, size_t chunk_) : chunk(chunk_), n(n_) { CKD(cuMemAddressReserve(&base, n * chunk, 0, 0, 0)); mp.assign(n, 0); }
    Range(const Range&) = delete; Range& operator=(const Range&) = delete;
    ~Range() { for (size_t i = 0; i < n; i++) if (mp[i]) cuMemUnmap(base + i * chunk, chunk); cuMemAddressFree(base, n * chunk); }
    void map(size_t i, CUmemGenericAllocationHandle h) { CKD(cuMemMap(base + i * chunk, chunk, 0, h, 0)); mp[i] = 1; }
    void access(size_t i, size_t cnt) { CUmemAccessDesc a = mkacc(); CKD(cuMemSetAccess(base + i * chunk, cnt * chunk, &a, 1)); }
    void unmap(size_t i) { CKD(cuMemUnmap(base + i * chunk, chunk)); mp[i] = 0; }
    void clear() { for (size_t i = 0; i < n; i++) if (mp[i]) unmap(i); }
    // unmap whatever is mapped, map handles H[hoff..hoff+n), one SetAccess for the whole range
    void set(const Handles& H, size_t hoff) { clear(); for (size_t i = 0; i < n; i++) map(i, H.h[hoff + i]); access(0, n); }
    char* ptr(size_t i = 0) const { return (char*)(uintptr_t)(base + i * chunk); }
};
// three kinds of 1 GiB device memory
static const char* KIND[3] = {"cudaMalloc", "vmm_2MiB_handles", "vmm_one_large_handle"};
struct Mem {
    void* plain = nullptr; Handles H; std::unique_ptr<Range> R;
    Mem(int kind, size_t sz) {
        if (kind == 0) { CK(cudaMalloc(&plain, sz)); return; }
        size_t chunk = kind == 1 ? CHUNK : sz;
        H.create(sz / chunk, chunk); R.reset(new Range(sz / chunk, chunk)); R->set(H, 0);
    }
    char* ptr() const { return plain ? (char*)plain : R->ptr(); }
    ~Mem() { if (plain) cudaFree(plain); }
};

// ---------- kernels ----------
__device__ __forceinline__ unsigned long long gtimer() { unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
__global__ void k_fill(unsigned* p, size_t n, unsigned seed) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = (unsigned)i * 2654435761u ^ seed;
}
__global__ void k_check(const unsigned* p, size_t n, unsigned seed, unsigned long long* mism) {
    unsigned long long m = 0;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) m += (p[i] != ((unsigned)i * 2654435761u ^ seed));
    if (m) atomicAdd(mism, m);
}
__global__ void k_sum32(const unsigned* p, size_t n, unsigned long long* out) {
    unsigned long long s = 0;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) s += p[i];
    atomicAdd(out, s);
}
// runs passes over p until `ns` nanoseconds elapsed; out_sum = sum of one full pass (summed over blocks), passes = total block-passes
__global__ void k_timed(const unsigned* p, size_t n, unsigned long long ns, unsigned long long* out_sum, unsigned long long* passes, volatile unsigned* started) {
    const volatile unsigned* vp = p; __shared__ int go;
    if (blockIdx.x == 0 && threadIdx.x == 0) { *started = 1; __threadfence_system(); }
    unsigned long long t0 = gtimer(), last = 0, np = 0;
    for (;;) {
        unsigned long long s = 0;
        for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) s += vp[i];
        last = s; np++;
        __syncthreads();
        if (threadIdx.x == 0) go = (gtimer() - t0) < ns;
        __syncthreads();
        if (!go) break;
    }
    atomicAdd(out_sum, last);
    if (threadIdx.x == 0) atomicAdd(passes, np);
}
__global__ void k_sum16(const uint4* p, size_t n, unsigned* out) {
    unsigned s = 0;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x, st = (size_t)gridDim.x * blockDim.x;
#pragma unroll 4
    for (; i < n; i += st) { uint4 x = p[i]; s += x.x + x.y + x.z + x.w; }
    atomicAdd(out, s);
}
__global__ void k_slots(const char* base, const size_t* offs, size_t slot_u4, unsigned* out) {
    const uint4* p = (const uint4*)(base + offs[blockIdx.y]); unsigned s = 0;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < slot_u4; i += (size_t)gridDim.x * blockDim.x) { uint4 x = p[i]; s += x.x + x.y + x.z + x.w; }
    atomicAdd(out, s);
}

static unsigned long long host_sum(size_t n, unsigned seed) {
    unsigned long long s = 0; for (size_t i = 0; i < n; i++) s += (unsigned)i * 2654435761u ^ seed; return s;
}

// ---------- 1 device ----------
static std::string sec_device() {
    CKD(cuInit(0)); CUdevice dev; CKD(cuDeviceGet(&dev, 0));
    char name[256] = {}; CKD(cuDeviceGetName(name, sizeof name, dev));
    int dv = 0, rv = 0, vmm = -1; cudaDriverGetVersion(&dv); cudaRuntimeGetVersion(&rv);
    CKD(cuDeviceGetAttribute(&vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, dev));
    size_t fr = 0, tot = 0; CK(cudaMemGetInfo(&fr, &tot));
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    std::string j = fmt("{\"worked\":true,\"name\":\"%s\",\"driver_version\":%d,\"runtime_version\":%d,\"total_bytes\":%zu,\"free_bytes\":%zu,"
                        "\"vmm_supported\":%d,\"sm_count\":%d,\"l2_bytes\":%d", name, dv, rv, tot, fr, vmm, p.multiProcessorCount, p.l2CacheSize);
    CUmemAllocationProp pr = mkprop(); size_t gmin = 0, grec = 0;
    try {
        CKD(cuMemGetAllocationGranularity(&gmin, &pr, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
        CKD(cuMemGetAllocationGranularity(&grec, &pr, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
        j += fmt(",\"granularity_min_bytes\":%zu,\"granularity_recommended_bytes\":%zu", gmin, grec);
    } catch (std::exception& x) { j += ",\"granularity\":" + jerr(x.what()); }
    return j + "}";
}

// ---------- 2 create ----------
static std::string sec_create() {
    CK(cudaDeviceSynchronize());
    size_t f0 = 0, f1 = 0, f2 = 0, f3 = 0, f4 = 0, tot = 0;
    CK(cudaMemGetInfo(&f0, &tot));
    Handles H; std::vector<double> per;
    double t0 = now_us();
    CUmemAllocationProp p = mkprop();
    for (int i = 0; i < NH; i++) {
        CUmemGenericAllocationHandle x = 0; double a = now_us(); CKD(cuMemCreate(&x, CHUNK, &p, 0)); per.push_back(now_us() - a); H.h.push_back(x);
    }
    double t_create = now_us() - t0;
    CK(cudaMemGetInfo(&f1, &tot));
    Range A(NH, CHUNK);
    double t1 = now_us();
    for (int i = 0; i < NH; i++) A.map(i, H.h[i]);
    double t_map = now_us() - t1; t1 = now_us(); A.access(0, NH); double t_acc = now_us() - t1;
    CK(cudaMemGetInfo(&f2, &tot));
    // touch to force residency, then re-query
    CK(cudaMemset(A.ptr(), 0, (size_t)NH * CHUNK)); CK(cudaDeviceSynchronize());
    CK(cudaMemGetInfo(&f3, &tot));
    t1 = now_us(); for (int i = 0; i < NH; i++) A.unmap(i); double t_unmap = now_us() - t1;
    t1 = now_us(); for (auto& x : H.h) { CKD(cuMemRelease(x)); x = 0; } double t_rel = now_us() - t1;
    CK(cudaMemGetInfo(&f4, &tot));
    auto d = [&](size_t a, size_t b) { return ((double)a - (double)b) / MiB; };  // MiB consumed going from a to b
    return fmt("{\"worked\":true,\"handles\":%d,\"create_total_ms\":%.3f,\"create_per_handle_us\":%s,\"map_all_ms\":%.3f,\"setaccess_all_one_call_ms\":%.3f,"
               "\"unmap_all_ms\":%.3f,\"release_all_ms\":%.3f,\"free_bytes\":{\"before_create\":%zu,\"after_create\":%zu,\"after_map_access\":%zu,\"after_memset_touch\":%zu,\"after_unmap_release\":%zu},"
               "\"consumed_MiB\":{\"at_create\":%.1f,\"at_map\":%.1f,\"at_touch\":%.1f,\"expected_total\":%.1f}}",
               NH, t_create / 1e3, st(per).c_str(), t_map / 1e3, t_acc / 1e3, t_unmap / 1e3, t_rel / 1e3, f0, f1, f2, f3, f4,
               d(f0, f1), d(f1, f2), d(f2, f3), (double)NH * CHUNK / MiB);
}

// ---------- 3 map_cost ----------
static std::string sec_map_cost() {
    Handles H; H.create(NH, CHUNK);
    Range A(NH, CHUNK), B(NH, CHUNK);
    std::string j = "{\"worked\":true";
    auto variant = [&](int batch, bool single_access, int reps) -> std::string {
        std::vector<double> m, a, u, ma;
        for (int r = 0; r < reps; r++)
            for (int s = 0; s < NH; s += batch) {
                int n = std::min(batch, NH - s);
                double t0 = now_us();
                for (int k = 0; k < n; k++) A.map(s + k, H.h[s + k]);
                double t1 = now_us();
                if (single_access) A.access(s, n); else for (int k = 0; k < n; k++) A.access(s + k, 1);
                double t2 = now_us();
                for (int k = 0; k < n; k++) A.unmap(s + k);
                double t3 = now_us();
                m.push_back((t1 - t0) / n); a.push_back((t2 - t1) / n); ma.push_back((t2 - t0) / n); u.push_back((t3 - t2) / n);
            }
        return fmt("{\"worked\":true,\"batch\":%d,\"setaccess\":\"%s\",\"samples_are_per_chunk_us_averaged_per_batch\":true,"
                   "\"map_us\":%s,\"setaccess_us\":%s,\"map_plus_setaccess_us\":%s,\"unmap_us\":%s}",
                   batch, single_access ? "one call per batch" : "one call per chunk", st(m).c_str(), st(a).c_str(), st(ma).c_str(), st(u).c_str());
    };
    add(j, "batch1_access_per_chunk", [&] { return variant(1, false, 3); });
    add(j, "batch60_access_per_chunk", [&] { return variant(60, false, 5); });
    add(j, "batch60_one_access_call", [&] { return variant(60, true, 5); });
    add(j, "batch1320_access_per_chunk", [&] { return variant(NH, false, 5); });
    add(j, "batch1320_one_access_call", [&] { return variant(NH, true, 5); });
    add(j, "event_120MiB", [&]() -> std::string {
        A.clear(); B.clear();
        for (int i = 0; i < NH; i++) A.map(i, H.h[i]);
        A.access(0, NH);
        Stream s; const int W = 60, NW = NH / W;
        std::vector<double> tot, tu, tm, ta, tz, p1, p2;
        for (int pass = 0; pass < 2; pass++) {
            Range& src = pass == 0 ? A : B; Range& dst = pass == 0 ? B : A;
            for (int w = 0; w < NW; w++) {
                int c0 = w * W;
                double t0 = now_us();
                for (int k = 0; k < W; k++) src.unmap(c0 + k);
                double t1 = now_us();
                for (int k = 0; k < W; k++) dst.map(c0 + k, H.h[c0 + k]);
                double t2 = now_us();
                dst.access(c0, W);
                double t3 = now_us();
                CK(cudaMemsetAsync(dst.ptr(c0), 0, (size_t)W * CHUNK, s)); CK(cudaStreamSynchronize(s));
                double t4 = now_us();
                tot.push_back((t4 - t0) / 1e3); tu.push_back((t1 - t0) / 1e3); tm.push_back((t2 - t1) / 1e3); ta.push_back((t3 - t2) / 1e3); tz.push_back((t4 - t3) / 1e3);
                (pass == 0 ? p1 : p2).push_back((t4 - t0) / 1e3);
            }
        }
        return fmt("{\"worked\":true,\"chunks_per_event\":%d,\"events\":%zu,\"total_ms\":%s,\"total_ms_pass1_A_to_B\":%s,\"total_ms_pass2_B_to_A\":%s,"
                   "\"unmap60_ms\":%s,\"map60_ms\":%s,\"setaccess_one_call_ms\":%s,\"memset120MiB_plus_sync_ms\":%s}",
                   W, tot.size(), st(tot).c_str(), st(p1).c_str(), st(p2).c_str(), st(tu).c_str(), st(tm).c_str(), st(ta).c_str(), st(tz).c_str());
    });
    return j + "}";
}

// ---------- 4 unmap_during_kernel ----------
static std::string sec_unmap_during_kernel() {
    const int KC = 16, AC = 8, NA = 4;       // kernel buffer 32 MiB; remap 4 chunks
    Handles HK, HA, HN; HK.create(KC, CHUNK); HA.create(AC, CHUNK);
    Range K(KC, CHUNK), A(AC, CHUNK), B(AC, CHUNK);
    K.set(HK, 0); A.set(HA, 0);
    size_t words = (size_t)KC * CHUNK / 4;
    CK(cudaMemset(K.ptr(), 1, (size_t)KC * CHUNK));          // each word 0x01010101
    unsigned long long expect = (unsigned long long)words * 0x01010101ull;
    DevBuf out(16); PinBuf flag(64, cudaHostAllocMapped); volatile unsigned* hf = (volatile unsigned*)flag.p;
    unsigned* df = nullptr; CK(cudaHostGetDevicePointer((void**)&df, flag.p, 0));
    Stream s; Event done;
    unsigned long long* o = (unsigned long long*)out.p;
    std::string j = "{\"worked\":true,\"kernel_target_ms\":50,\"remapped_chunks\":4";
    auto launch = [&](double& t_launch) {
        CK(cudaMemset(out.p, 0, 16)); *hf = 0;
        t_launch = now_us();
        k_timed<<<32, 256, 0, s>>>((const unsigned*)K.ptr(), words, 50000000ull, o, o + 1, df);
        CK(cudaGetLastError()); CK(cudaEventRecord(done.e, s));
    };
    auto check = [&](bool& ok, unsigned long long& passes) {
        unsigned long long h[2]; CK(cudaMemcpy(h, out.p, 16, cudaMemcpyDeviceToHost));
        ok = h[0] == expect; passes = h[1];
    };
    // control
    add(j, "control_no_ops", [&]() -> std::string {
        std::vector<double> w;
        for (int i = 0; i < 3; i++) { double tl; launch(tl); CK(cudaStreamSynchronize(s)); w.push_back((now_us() - tl) / 1e3); }
        bool ok; unsigned long long ps; check(ok, ps);
        return fmt("{\"worked\":true,\"kernel_wall_ms\":%s,\"result_correct\":%s,\"block_passes\":%llu}", st(w).c_str(), ok ? "true" : "false", ps);
    });
    std::string trials = "[";
    for (int t = 0; t < 3; t++) {
        std::string r;
        try {
            double tl; launch(tl);
            double tw = now_us();
            while (*hf != 1) { if (now_us() - tw > 2e6) throw std::runtime_error("kernel never signalled start"); }
            double t0 = now_us();
            for (int k = 0; k < NA; k++) A.unmap(k);
            double t1 = now_us();
            for (int k = 0; k < NA; k++) B.map(k, HA.h[k]);
            double t2 = now_us(); B.access(0, NA); double t3 = now_us();
            // also create/release a fresh handle while the kernel runs
            CUmemAllocationProp p = mkprop(); CUmemGenericAllocationHandle nh = 0;
            double t4 = now_us(); CUresult cr = cuMemCreate(&nh, CHUNK, &p, 0); double t5 = now_us();
            if (cr == CUDA_SUCCESS) cuMemRelease(nh);
            cudaError_t q = cudaEventQuery(done.e);
            bool still_running = (q == cudaErrorNotReady);
            CK(cudaStreamSynchronize(s)); double tend = now_us();
            bool ok; unsigned long long ps; check(ok, ps);
            // restore for next trial
            B.clear(); for (int k = 0; k < NA; k++) A.map(k, HA.h[k]); A.access(0, NA);
            r = fmt("{\"worked\":true,\"start_signal_wait_ms\":%.3f,\"unmap4_call_ms\":%.3f,\"map4_call_ms\":%.3f,\"setaccess4_call_ms\":%.3f,\"create1_call_ms\":%.3f,\"create1_result\":%d,"
                    "\"kernel_still_running_when_calls_returned\":%s,\"kernel_wall_ms\":%.3f,\"result_correct\":%s,\"block_passes\":%llu}",
                    (tw - tl) / 1e3, (t1 - t0) / 1e3, (t2 - t1) / 1e3, (t3 - t2) / 1e3, (t5 - t4) / 1e3, (int)cr,
                    still_running ? "true" : "false", (tend - tl) / 1e3, ok ? "true" : "false", ps);
        } catch (std::exception& x) { cudaGetLastError(); cudaStreamSynchronize(s); r = jerr(x.what()); }
        trials += (t ? "," : "") + r;
    }
    return j + ",\"trials\":" + trials + "]}";
}

// ---------- 5 remap_correctness ----------
static std::string sec_remap_correctness() {
    const int N = 4; const size_t words = (size_t)N * CHUNK / 4;
    const unsigned S1 = 0xA5A5u, S2 = 0x5C3Du;
    Handles H; H.create(8, CHUNK);            // H[0..3] pattern 1, H[4..7] pattern 2
    Range A(N, CHUNK), B(N, CHUNK), C(N, CHUNK);
    DevBuf mm(8), sum(8); PinBuf hs(64); Stream s;
    auto fill = [&](Range& R, unsigned seed) { k_fill<<<256, 256, 0, s>>>((unsigned*)R.ptr(), words, seed); CK(cudaGetLastError()); CK(cudaStreamSynchronize(s)); };
    auto check = [&](Range& R, unsigned seed) -> unsigned long long {
        unsigned long long m = 0; CK(cudaMemset(mm.p, 0, 8));
        k_check<<<256, 256, 0, s>>>((const unsigned*)R.ptr(), words, seed, (unsigned long long*)mm.p); CK(cudaGetLastError()); CK(cudaStreamSynchronize(s));
        CK(cudaMemcpy(&m, mm.p, 8, cudaMemcpyDeviceToHost)); return m;
    };
    std::string j = "{\"worked\":true";
    add(j, "preserved_across_unmap_and_map_elsewhere", [&]() -> std::string {
        A.set(H, 0); fill(A, S1); A.clear(); B.set(H, 0);
        unsigned long long m = check(B, S1);
        return fmt("{\"worked\":true,\"mismatching_words\":%llu,\"words\":%zu,\"contents_preserved\":%s}", m, words, m == 0 ? "true" : "false");
    });
    add(j, "new_handles_at_same_va", [&]() -> std::string {
        B.clear(); C.set(H, 4); fill(C, S2); C.clear(); A.set(H, 4);
        unsigned long long m2 = check(A, S2), m1 = check(A, S1);
        return fmt("{\"worked\":true,\"va\":\"range A (same address as before)\",\"mismatches_vs_new_pattern\":%llu,\"mismatches_vs_old_pattern\":%llu,\"kernel_sees_new_contents\":%s}",
                   m2, m1, (m2 == 0 && m1 > 0) ? "true" : "false");
    });
    add(j, "cuda_graph_fixed_va", [&]() -> std::string {
        C.clear(); B.clear();
        A.set(H, 4);                                   // pattern 2 at A
        unsigned long long e2 = host_sum(words, S2), e1 = host_sum(words, S1);
        Graph G; unsigned long long* hv = (unsigned long long*)hs.p; unsigned long long* dv = (unsigned long long*)sum.p;
        CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
        try {
            CK(cudaMemsetAsync(dv, 0, 8, s));
            k_sum32<<<256, 256, 0, s>>>((const unsigned*)A.ptr(), words, dv); CK(cudaGetLastError());
            CK(cudaMemcpyAsync(hv, dv, 8, cudaMemcpyDeviceToHost, s));
        } catch (...) { cudaGraph_t g; cudaStreamEndCapture(s, &g); if (g) cudaGraphDestroy(g); throw; }
        CK(cudaStreamEndCapture(s, &G.g));
        CK(cudaGraphInstantiate(&G.x, G.g, 0));
        *hv = 0; CK(cudaGraphLaunch(G.x, s)); CK(cudaStreamSynchronize(s)); unsigned long long r1 = *hv;
        A.set(H, 0);                                   // remap pattern 1 at the SAME address
        *hv = 0; CK(cudaGraphLaunch(G.x, s)); CK(cudaStreamSynchronize(s)); unsigned long long r2 = *hv;
        A.set(H, 4);                                   // and back
        *hv = 0; CK(cudaGraphLaunch(G.x, s)); CK(cudaStreamSynchronize(s)); unsigned long long r3 = *hv;
        bool a = r1 == e2, b = r2 == e1, c = r3 == e2;
        return fmt("{\"worked\":true,\"graph_first_launch_correct\":%s,\"graph_after_remap_sees_new_data\":%s,\"graph_after_remap_back_correct\":%s,"
                   "\"sums\":[%llu,%llu,%llu],\"expected\":[%llu,%llu,%llu]}",
                   a ? "true" : "false", b ? "true" : "false", c ? "true" : "false", r1, r2, r3, e2, e1, e2);
    });
    return j + "}";
}

// ---------- 6 read_bandwidth ----------
static std::string bw_json(const std::function<void()>& launch, double bytes, cudaStream_t s) {
    Event a, b; std::vector<double> ms;
    for (int r = 0; r < 18; r++) {
        CK(cudaEventRecord(a.e, s)); launch(); CK(cudaGetLastError()); CK(cudaEventRecord(b.e, s)); CK(cudaEventSynchronize(b.e));
        if (r >= 3) ms.push_back(ev_ms(a, b));
    }
    return fmt("{\"worked\":true,\"bytes\":%.0f,\"ms\":%s,\"GBps_median\":%.2f,\"GBps_best\":%.2f}", bytes, st(ms).c_str(), bytes / (med(ms) * 1e6), bytes / (mn(ms) * 1e6));
}
static std::string sec_read_bandwidth() {
    std::string j = "{\"worked\":true,\"note\":\"1 GiB region exceeds L2; slots = 141 x 2034240 B at fixed random offsets (same for all kinds)\"";
    int sms = 0; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    for (int k = 0; k < 3; k++)
        add(j, KIND[k], [&]() -> std::string {
            Mem m(k, GiB); Stream s; DevBuf out(8), offs(NSLOT * sizeof(size_t));
            CK(cudaMemset(m.ptr(), 1, GiB)); CK(cudaMemset(out.p, 0, 8));
            CK(cudaMemcpy(offs.p, g_offs.data(), NSLOT * sizeof(size_t), cudaMemcpyHostToDevice));
            CK(cudaDeviceSynchronize());
            std::string r = "{\"worked\":true,\"sequential_1GiB\":";
            r += bw_json([&] { k_sum16<<<sms * 8, 256, 0, s>>>((const uint4*)m.ptr(), GiB / 16, (unsigned*)out.p); }, (double)GiB, s);
            r += ",\"random_141_slots_1.94MiB\":";
            r += bw_json([&] { k_slots<<<dim3(32, NSLOT), 256, 0, s>>>(m.ptr(), (const size_t*)offs.p, SLOT / 16, (unsigned*)out.p); }, (double)NSLOT * SLOT, s);
            return r + "}";
        });
    return j + "}";
}

// ---------- 7 h2d ----------
static std::string timed(int count, double bytes, cudaStream_t s, const std::function<void()>& issue) {
    std::vector<double> ti, tc;
    for (int r = 0; r < 13; r++) {
        double t0 = now_us(); issue(); double t1 = now_us(); CK(cudaStreamSynchronize(s)); double t2 = now_us();
        if (r >= 3) { ti.push_back(t1 - t0); tc.push_back(t2 - t0); }
    }
    return fmt("{\"worked\":true,\"copies\":%d,\"bytes\":%.0f,\"issue_us\":%s,\"completion_us\":%s,\"GBps_median\":%.2f,\"GBps_best\":%.2f}",
               count, bytes, st(ti).c_str(), st(tc).c_str(), bytes / (med(tc) * 1e3), bytes / (mn(tc) * 1e3));
}
static std::string sec_h2d() {
    PinBuf h(GiB); memset(h.p, 1, GiB);
    std::string j = "{\"worked\":true,\"note\":\"pinned host source; 141-copy destinations are the random slot offsets; src slots contiguous in the pinned buffer\"";
    for (int k = 0; k < 3; k++)
        add(j, KIND[k], [&]() -> std::string {
            Mem m(k, GiB); Stream s; char* dp = m.ptr(); char* hp = (char*)h.p;
            CK(cudaMemset(dp, 0, GiB)); CK(cudaDeviceSynchronize());
            std::string r = "{\"worked\":true";
            add(r, "one_copy_1GiB", [&] { return timed(1, (double)GiB, s, [&] { CK(cudaMemcpyAsync(dp, hp, GiB, cudaMemcpyHostToDevice, s)); }); });
            add(r, "individual_141x1.94MiB", [&] {
                return timed(NSLOT, (double)NSLOT * SLOT, s, [&] { for (int i = 0; i < NSLOT; i++) CK(cudaMemcpyAsync(dp + g_offs[i], hp + (size_t)i * SLOT, SLOT, cudaMemcpyHostToDevice, s)); }); });
            add(r, "batch_141x1.94MiB_cudaMemcpyBatchAsync", [&] {
                std::vector<void*> D(NSLOT), S(NSLOT); std::vector<size_t> Z(NSLOT, SLOT);
                for (int i = 0; i < NSLOT; i++) { D[i] = dp + g_offs[i]; S[i] = hp + (size_t)i * SLOT; }
                cudaMemcpyAttributes at = {}; at.srcAccessOrder = cudaMemcpySrcAccessOrderStream; size_t idx = 0;
                return timed(NSLOT, (double)NSLOT * SLOT, s, [&] { CK(cudaMemcpyBatchAsync(D.data(), S.data(), Z.data(), NSLOT, &at, &idx, 1, s)); }); });
            add(r, "d2d_100x1.94MiB_inside_memory", [&] {
                return timed(100, 100.0 * SLOT, s, [&] { for (int i = 0; i < 100; i++) CK(cudaMemcpyAsync(dp + g_offs[(i + 41) % NSLOT], dp + g_offs[i], SLOT, cudaMemcpyDeviceToDevice, s)); }); });
            return r + "}";
        });
    return j + "}";
}

// ---------- 8 release ----------
static std::string sec_release() {
    CK(cudaDeviceSynchronize());
    size_t fr = 0, tot = 0; CK(cudaMemGetInfo(&fr, &tot));
    return fmt("{\"worked\":true,\"free_bytes_start\":%zu,\"free_bytes_end\":%zu,\"leak_bytes\":%lld,\"leak_MiB\":%.2f}",
               g_start_free, fr, (long long)g_start_free - (long long)fr, ((double)g_start_free - (double)fr) / MiB);
}

int main() {
    std::string out = "{";
    cudaSetDeviceFlags(cudaDeviceMapHost); cudaGetLastError();
    try {
        CK(cudaSetDevice(0)); CK(cudaFree(0)); CK(cudaMemGetInfo(&g_start_free, &g_total));
    } catch (std::exception& x) { fprintf(stderr, "init failed: %s\n", x.what()); }
    {
        size_t n = GiB / SLOT; std::vector<size_t> idx(n); for (size_t i = 0; i < n; i++) idx[i] = i;
        std::mt19937 rng(12345); std::shuffle(idx.begin(), idx.end(), rng);
        for (int i = 0; i < NSLOT; i++) g_offs.push_back(idx[i] * SLOT);
    }
    struct { const char* key; std::string (*fn)(); } secs[] = {
        {"device", sec_device}, {"create", sec_create}, {"map_cost", sec_map_cost}, {"unmap_during_kernel", sec_unmap_during_kernel},
        {"remap_correctness", sec_remap_correctness}, {"read_bandwidth", sec_read_bandwidth}, {"h2d", sec_h2d}, {"release", sec_release}};
    for (auto& s : secs) add(out, s.key, [&] { return s.fn(); }), cudaGetLastError(), cudaDeviceSynchronize();
    out += "}";
    printf("%s\n", out.c_str());
    cudaDeviceReset();
    return 0;
}
