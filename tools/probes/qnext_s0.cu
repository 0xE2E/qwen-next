// qnext_s0: standalone CUDA micro-benchmark for launch / sync / copy / mailbox costs.
// JSON on stdout, progress on stderr. Each experiment is independent and records
// {"supported": false, "error": "..."} on failure.
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdarg>
#include <atomic>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using clk = std::chrono::steady_clock;
static double now_us() {
    return std::chrono::duration<double, std::micro>(clk::now().time_since_epoch()).count();
}
static std::string fmt(const char* f, ...) {
    char b[1024]; va_list a; va_start(a, f); vsnprintf(b, sizeof b, f, a); va_end(a); return b;
}
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) \
    throw std::runtime_error(fmt("%s failed: %s (%s:%d)", #x, cudaGetErrorString(e_), __FILE__, __LINE__)); } while (0)
#define CKD(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char* s_ = "?"; cuGetErrorString(r_, &s_); \
    throw std::runtime_error(fmt("%s failed: %s (%d)", #x, s_, (int)r_)); } } while (0)

// ---------- stats / json ----------
static std::string st(std::vector<double> v) {
    if (v.empty()) return "null";
    std::sort(v.begin(), v.end());
    return fmt("{\"min\":%.3f,\"median\":%.3f,\"p90\":%.3f,\"n\":%zu}", v.front(), v[v.size() / 2],
               v[std::min(v.size() - 1, (size_t)(v.size() * 0.9))], v.size());
}
static double med(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
static std::string jerr(const std::string& m) {
    std::string e; for (char c : m) { if (c == '"' || c == '\\') e += '\\'; if (c == '\n') c = ' '; e += c; }
    return "{\"supported\":false,\"error\":\"" + e + "\"}";
}

// ---------- RAII ----------
struct Stream { cudaStream_t s = nullptr; Stream() { CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)); }
    ~Stream() { if (s) { cudaStreamSynchronize(s); cudaStreamDestroy(s); } } operator cudaStream_t() const { return s; } };
struct DevBuf { void* p = nullptr; explicit DevBuf(size_t n) { CK(cudaMalloc(&p, n)); } ~DevBuf() { cudaFree(p); } };
struct PinBuf { void* p = nullptr; explicit PinBuf(size_t n, unsigned fl = cudaHostAllocDefault) { CK(cudaHostAlloc(&p, n, fl)); } ~PinBuf() { if (p) cudaFreeHost(p); } };
struct Event { cudaEvent_t e = nullptr; Event() { CK(cudaEventCreate(&e)); } ~Event() { cudaEventDestroy(e); } };
struct Graph { cudaGraph_t g = nullptr; cudaGraphExec_t x = nullptr;
    ~Graph() { if (x) cudaGraphExecDestroy(x); if (g) cudaGraphDestroy(g); } };

// ---------- kernels ----------
__global__ void k_empty() {}
__global__ void k_small(float* x) { int i = blockIdx.x * 256 + threadIdx.x; x[i] = x[i] * 0.999f + 0.001f; }
__global__ void k_empty_pdl(int) { cudaGridDependencySynchronize(); cudaTriggerProgrammaticLaunchCompletion(); }
__global__ void k_small_pdl(float* x) {
    cudaGridDependencySynchronize(); cudaTriggerProgrammaticLaunchCompletion();
    int i = blockIdx.x * 256 + threadIdx.x; x[i] = x[i] * 0.999f + 0.001f;
}
__global__ void k_sum(const float* x, float* out, int n) {
    float s = 0; for (int i = blockIdx.x * 256 + threadIdx.x; i < n; i += gridDim.x * 256) s += x[i];
    __shared__ float sh[256]; sh[threadIdx.x] = s; __syncthreads();
    for (int o = 128; o; o >>= 1) { if (threadIdx.x < o) sh[threadIdx.x] += sh[threadIdx.x + o]; __syncthreads(); }
    if (threadIdx.x == 0) atomicAdd(out, sh[0]);
}
__global__ void k_store(volatile unsigned* p, unsigned v) { *p = v; __threadfence_system(); }
// spin on flag until == target or cap cycles elapse; ack = v on success, 0xFFFFFFFF on timeout
__global__ void k_spin(volatile unsigned* flag, unsigned target, volatile unsigned* started, unsigned sv,
                       volatile unsigned* ack, long long cap_cycles, bool announce) {
    if (threadIdx.x) return;
    if (announce) { *started = sv; __threadfence_system(); }
    long long t0 = clock64(); bool ok = false;
    while (clock64() - t0 < cap_cycles) { if (*flag == target) { ok = true; break; } }
    *ack = ok ? target : 0xFFFFFFFFu; __threadfence_system();
}

static const int NSMALL = 10240;
static const int NSUM = 5000000;

// ---------- chain launching helpers ----------
enum Kind { EMPTY, SMALL, MIXED };
struct Ctx { float* buf; float* big; float* out; };
static void launch_one(Kind k, int i, Ctx& c, cudaStream_t s) {
    if (k == EMPTY) k_empty<<<1, 1, 0, s>>>();
    else if (k == MIXED && i % 14 == 0) k_sum<<<160, 256, 0, s>>>(c.big, c.out, NSUM);
    else k_small<<<40, 256, 0, s>>>(c.buf);
}
static void launch_chain(Kind k, int n, Ctx& c, cudaStream_t s, bool pdl) {
    if (!pdl) { for (int i = 0; i < n; i++) launch_one(k, i, c, s); CK(cudaGetLastError()); return; }
    cudaLaunchAttribute at; at.id = cudaLaunchAttributeProgrammaticStreamSerialization; at.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg = {}; cfg.stream = s; cfg.attrs = &at; cfg.numAttrs = 1;
    for (int i = 0; i < n; i++) {
        if (k == EMPTY) { cfg.gridDim = dim3(1); cfg.blockDim = dim3(1); CK(cudaLaunchKernelEx(&cfg, k_empty_pdl, 0)); }
        else { cfg.gridDim = dim3(40); cfg.blockDim = dim3(256); CK(cudaLaunchKernelEx(&cfg, k_small_pdl, c.buf)); }
    }
}
static const int REPS = 30, WARM = 5;

static std::string direct_stats(Kind k, int n, Ctx& c, cudaStream_t s) {
    std::vector<double> tl, tw;
    for (int r = 0; r < REPS + WARM; r++) {
        double t0 = now_us(); launch_chain(k, n, c, s, false); double t1 = now_us();
        CK(cudaStreamSynchronize(s)); double t2 = now_us();
        if (r >= WARM) { tl.push_back(t1 - t0); tw.push_back(t2 - t0); }
    }
    return fmt("{\"launch_total_us\":%s,\"wall_total_us\":%s,\"launch_per_kernel_us\":%.4f,\"wall_per_kernel_us\":%.4f}",
               st(tl).c_str(), st(tw).c_str(), med(tl) / n, med(tw) / n);
}
static std::string graph_stats(Kind k, int n, Ctx& c, cudaStream_t s, bool pdl) {
    Graph G;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    try { launch_chain(k, n, c, s, pdl); } catch (...) { cudaGraph_t g; cudaStreamEndCapture(s, &g); if (g) cudaGraphDestroy(g); throw; }
    CK(cudaStreamEndCapture(s, &G.g));
    double t0 = now_us(); CK(cudaGraphInstantiate(&G.x, G.g, 0)); double inst = now_us() - t0;
    std::vector<double> tl, tw;
    for (int r = 0; r < REPS + WARM; r++) {
        double a = now_us(); CK(cudaGraphLaunch(G.x, s)); double b = now_us();
        CK(cudaStreamSynchronize(s)); double d = now_us();
        if (r >= WARM) { tl.push_back(b - a); tw.push_back(d - a); }
    }
    return fmt("{\"instantiate_us\":%.1f,\"launch_call_us\":%s,\"wall_per_launch_us\":%s,\"wall_per_kernel_us\":%.4f}",
               inst, st(tl).c_str(), st(tw).c_str(), med(tw) / n);
}

static std::string exp_A() {
    DevBuf buf(NSMALL * 4), big((size_t)NSUM * 4), out(4);
    CK(cudaMemset(buf.p, 0, NSMALL * 4)); CK(cudaMemset(big.p, 0, (size_t)NSUM * 4)); CK(cudaMemset(out.p, 0, 4));
    Ctx c{(float*)buf.p, (float*)big.p, (float*)out.p};
    Stream s;
    std::string j = "{";
    const int Ns[] = {1, 100, 700, 2500};
    struct { const char* name; Kind k; } kinds[] = {{"empty", EMPTY}, {"small", SMALL}};
    for (auto& kd : kinds) {
        j += fmt("\"%s_kernel\":{", kd.name); bool first = true;
        for (int n : Ns) {
            std::string e;
            try { e = fmt("{\"direct\":%s,\"graph\":", direct_stats(kd.k, n, c, s).c_str());
                  e += graph_stats(kd.k, n, c, s, false) + ",\"graph_pdl\":";
                  try { e += graph_stats(kd.k, n, c, s, true); } catch (std::exception& x) { cudaGetLastError(); e += jerr(x.what()); }
                  e += "}"; }
            catch (std::exception& x) { cudaGetLastError(); e = jerr(x.what()); }
            fprintf(stderr, "[A] %s N=%d done\n", kd.name, n);
            j += fmt("%s\"N%d\":%s", first ? "" : ",", n, e.c_str()); first = false;
        }
        j += "},";
    }
    try { j += "\"mixed_700\":{\"direct\":" + direct_stats(MIXED, 700, c, s) + ",\"graph\":" + graph_stats(MIXED, 700, c, s, false) + "}"; }
    catch (std::exception& x) { cudaGetLastError(); j += "\"mixed_700\":" + jerr(x.what()); }
    return j + "}";
}

// ---------- B ----------
static std::string exp_B() {
    Stream s; DevBuf d(4096), kb(NSMALL * 4); PinBuf h(4096);
    CK(cudaMemset(kb.p, 0, NSMALL * 4));
    std::vector<double> idle, d2h, h2d;
    for (int r = 0; r < 220; r++) {
        double t0 = now_us(); CK(cudaStreamSynchronize(s)); double t1 = now_us();
        k_small<<<40, 256, 0, s>>>((float*)kb.p); CK(cudaMemcpyAsync(h.p, d.p, 40, cudaMemcpyDeviceToHost, s)); CK(cudaStreamSynchronize(s)); double t2 = now_us();
        CK(cudaMemcpyAsync(d.p, h.p, 2048, cudaMemcpyHostToDevice, s)); CK(cudaStreamSynchronize(s)); double t3 = now_us();
        if (r >= 20) { idle.push_back(t1 - t0); d2h.push_back(t2 - t1); h2d.push_back(t3 - t2); }
    }
    return "{\"idle_stream_sync_us\":" + st(idle) + ",\"kernel_plus_d2h40B_plus_sync_us\":" + st(d2h) + ",\"h2d2KB_plus_sync_us\":" + st(h2d) + "}";
}

// ---------- C ----------
static const size_t TOTAL = 273000000;
static std::string copy_mode(const char* name, int count, void* hp, void* dp, cudaStream_t s,
                             const std::function<void()>& issue) {
    std::vector<double> ti, tc;
    for (int r = 0; r < 18; r++) {
        double t0 = now_us(); issue(); double t1 = now_us(); CK(cudaStreamSynchronize(s)); double t2 = now_us();
        if (r >= 3) { ti.push_back(t1 - t0); tc.push_back(t2 - t0); }
    }
    (void)hp; (void)dp;
    return fmt("\"%s\":{\"copies\":%d,\"issue_us\":%s,\"completion_us\":%s,\"GBps_median\":%.3f}", name, count, st(ti).c_str(), st(tc).c_str(),
               TOTAL / (med(tc) * 1e3));
}
static std::string exp_C() {
    PinBuf h(TOTAL); DevBuf d(TOTAL); Stream s;
    memset(h.p, 1, TOTAL);
    char* hp = (char*)h.p; char* dp = (char*)d.p;
    std::string j = "{";
    auto multi = [&](int n) { size_t sz = TOTAL / n; for (int i = 0; i < n; i++) CK(cudaMemcpyAsync(dp + i * sz, hp + i * sz, sz, cudaMemcpyHostToDevice, s)); };
    j += copy_mode("n420_x650KB", 420, hp, dp, s, [&] { multi(420); }) + ",";
    j += copy_mode("n140_x1950KB", 140, hp, dp, s, [&] { multi(140); }) + ",";
    j += copy_mode("n1_x273MB", 1, hp, dp, s, [&] { multi(1); }) + ",";
    try {
        const int n = 420; size_t sz = TOTAL / n;
        std::vector<void*> D(n), S(n); std::vector<size_t> Z(n, sz);
        for (int i = 0; i < n; i++) { D[i] = dp + i * sz; S[i] = hp + i * sz; }
        cudaMemcpyAttributes at = {}; at.srcAccessOrder = cudaMemcpySrcAccessOrderStream; size_t idx = 0;
        j += copy_mode("batch420_x650KB", n, hp, dp, s, [&] { CK(cudaMemcpyBatchAsync(D.data(), S.data(), Z.data(), n, &at, &idx, 1, s)); });
    } catch (std::exception& x) { cudaGetLastError(); j += "\"batch420_x650KB\":" + jerr(x.what()); }
    return j + "}";
}

// ---------- D ----------
static std::string exp_D() {
    const int n = 420; const size_t sz = 650000;
    PinBuf h(n * sz); DevBuf d(n * sz), buf(NSMALL * 4); Stream s1, s2;
    memset(h.p, 1, n * sz); CK(cudaMemset(buf.p, 0, NSMALL * 4));
    Ctx c{(float*)buf.p, nullptr, nullptr};
    std::atomic<bool> stop{false}, running{false}; std::atomic<double> wbytes{0}, wsecs{0}; std::string werr;
    auto worker = [&](double min_secs) {
        try {
            double t0 = now_us(), bytes = 0; running = true;
            while (!stop.load() || now_us() - t0 < min_secs * 1e6 * 0) {
                for (int i = 0; i < n; i++) CK(cudaMemcpyAsync((char*)d.p + i * sz, (char*)h.p + i * sz, sz, cudaMemcpyHostToDevice, s2));
                CK(cudaStreamSynchronize(s2)); bytes += (double)n * sz;
            }
            wbytes = bytes; wsecs = (now_us() - t0) * 1e-6;
        } catch (std::exception& e) { werr = e.what(); running = true; }
    };
    auto main_run = [&](std::vector<double>& out) {
        for (int r = 0; r < REPS + WARM; r++) {
            double t0 = now_us(); launch_chain(SMALL, 700, c, s1, false); CK(cudaStreamSynchronize(s1));
            if (r >= WARM) out.push_back(now_us() - t0);
        }
    };
    std::vector<double> alone, with;
    main_run(alone);
    // worker alone for ~0.4 s
    stop = false; running = false; std::thread t(worker, 0.0);
    while (!running) std::this_thread::yield();
    std::this_thread::sleep_for(std::chrono::milliseconds(400)); stop = true; t.join();
    if (!werr.empty()) throw std::runtime_error(werr);
    double w_alone = wbytes / wsecs / 1e9;
    // together
    stop = false; running = false; t = std::thread(worker, 0.0);
    while (!running) std::this_thread::yield();
    main_run(with); stop = true; t.join();
    if (!werr.empty()) throw std::runtime_error(werr);
    double w_with = wbytes / wsecs / 1e9;
    return "{\"main_chain700_wall_us_alone\":" + st(alone) + ",\"main_chain700_wall_us_with_worker\":" + st(with) +
           fmt(",\"worker_GBps_alone\":%.3f,\"worker_GBps_with_main\":%.3f,\"main_slowdown_median\":%.3f}", w_alone, w_with, med(with) / med(alone));
}

// ---------- E ----------
static inline unsigned rd(volatile unsigned* p) { return *p; }
// spin until *p==v, return elapsed us or -1 on 2 s timeout
static double spin_for(volatile unsigned* p, unsigned v, double t0) {
    for (;;) { if (rd(p) == v) return now_us() - t0; if (now_us() - t0 > 2e6) return -1; }
}
static std::string mailbox_mode(const char* name, bool mapped) {
    unsigned* hp = nullptr;
    if (mapped) CK(cudaHostAlloc((void**)&hp, 4096, cudaHostAllocMapped)); else CK(cudaMallocHost((void**)&hp, 4096));
    struct Fr { unsigned* p; ~Fr() { cudaFreeHost(p); } } fr{hp};
    memset(hp, 0, 4096);
    unsigned* dp = hp;
    if (mapped) CK(cudaHostGetDevicePointer((void**)&dp, hp, 0));
    volatile unsigned *mb = hp, *flag = hp + 16, *ack = hp + 32, *started = hp + 48;
    unsigned *dmb = dp, *dflag = dp + 16, *dack = dp + 32, *dstarted = dp + 48;
    Stream s, s2; std::string j = "{";
    int clk_khz = 0; CK(cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, 0));
    long long cap = (long long)clk_khz * 200; // ~200 ms in cycles
    // 1: device -> host
    {
        std::vector<double> lat; bool ok = true; std::string why;
        for (unsigned i = 1; i <= 200; i++) {
            k_store<<<1, 1, 0, s>>>(dmb, i); CK(cudaGetLastError());
            double t0 = now_us(); double l = spin_for(mb, i, t0);
            if (l < 0) { ok = false; why = fmt("host did not observe kernel write within 2 s at trial %u", i); CK(cudaStreamSynchronize(s)); break; }
            lat.push_back(l);
        }
        CK(cudaStreamSynchronize(s));
        j += fmt("\"kernel_to_host\":{\"worked\":%s,\"error\":\"%s\",\"latency_from_launch_return_us\":%s}", ok ? "true" : "false", why.c_str(), st(lat).c_str());
    }
    // 2: host -> running kernel
    {
        std::vector<double> lat; bool ok = true; std::string why;
        for (unsigned i = 1; i <= 100; i++) {
            k_spin<<<1, 32, 0, s>>>(dflag, i, dstarted, i, dack, cap, true); CK(cudaGetLastError());
            if (spin_for(started, i, now_us()) < 0) { ok = false; why = "kernel never announced start"; break; }
            double t0 = now_us(); *flag = i; std::atomic_thread_fence(std::memory_order_seq_cst);
            double l = spin_for(ack, i, t0);
            if (l < 0 || rd(ack) != i) { ok = false; why = fmt("kernel did not see host flag write at trial %u (ack=0x%x)", i, rd(ack)); CK(cudaStreamSynchronize(s)); break; }
            lat.push_back(l); CK(cudaStreamSynchronize(s));
        }
        CK(cudaStreamSynchronize(s));
        j += fmt(",\"host_to_running_kernel\":{\"worked\":%s,\"error\":\"%s\",\"flag_write_to_ack_observed_us\":%s}", ok ? "true" : "false", why.c_str(), st(lat).c_str());
    }
    // 3: round trip with 1.95 MB H2D copy between
    {
        const size_t sz = 1950000; PinBuf hc(sz); DevBuf dc(sz); memset(hc.p, 2, sz);
        std::vector<double> rt, from_launch; bool ok = true; std::string why;
        unsigned base = 1000;
        for (unsigned i = 1; i <= 100; i++) {
            unsigned v = base + i;
            k_store<<<1, 1, 0, s>>>(dmb, v);
            k_spin<<<1, 32, 0, s>>>(dflag, v, dstarted, v, dack, cap, false);   // ack==v when released
            k_store<<<1, 1, 0, s>>>(dstarted, v);                                // following kernel
            CK(cudaGetLastError());
            double tl = now_us();
            double a = spin_for(mb, v, tl);
            if (a < 0) { ok = false; why = "mailbox not observed"; CK(cudaStreamSynchronize(s)); break; }
            double tseen = now_us();
            CK(cudaMemcpyAsync(dc.p, hc.p, sz, cudaMemcpyHostToDevice, s2)); CK(cudaStreamSynchronize(s2));
            *flag = v; std::atomic_thread_fence(std::memory_order_seq_cst);
            double l = spin_for(started, v, tseen);
            double tdone = now_us();
            if (l < 0 || rd(ack) != v) { ok = false; why = fmt("round trip failed at trial %u (ack=0x%x)", i, rd(ack)); CK(cudaStreamSynchronize(s)); break; }
            rt.push_back(tdone - tseen); from_launch.push_back(tdone - tl); CK(cudaStreamSynchronize(s));
        }
        CK(cudaStreamSynchronize(s));
        j += fmt(",\"round_trip_1p95MB\":{\"worked\":%s,\"error\":\"%s\",\"mailbox_seen_to_following_kernel_observed_us\":%s,\"launch_return_to_following_kernel_observed_us\":%s}",
                 ok ? "true" : "false", why.c_str(), st(rt).c_str(), st(from_launch).c_str());
    }
    (void)dack; (void)dstarted;
    return j + "}";
}
static std::string exp_E() {
    cudaSetDeviceFlags(cudaDeviceMapHost); cudaGetLastError(); // may be a no-op once the context exists
    std::string j = "{";
    try { j += "\"cudaHostAllocMapped\":" + mailbox_mode("mapped", true); } catch (std::exception& x) { cudaGetLastError(); j += "\"cudaHostAllocMapped\":" + jerr(x.what()); }
    try { j += ",\"cudaMallocHost_same_pointer\":" + mailbox_mode("plain", false); } catch (std::exception& x) { cudaGetLastError(); j += ",\"cudaMallocHost_same_pointer\":" + jerr(x.what()); }
    return j + "}";
}

// ---------- F ----------
static std::string exp_F() {
    CKD(cuInit(0)); CUdevice dev; CKD(cuDeviceGet(&dev, 0));
    int nor = 0, v1 = 0, b64 = 0;
    CKD(cuDeviceGetAttribute(&nor, CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR, dev));
    CKD(cuDeviceGetAttribute(&v1, CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1, dev));
    CKD(cuDeviceGetAttribute(&b64, CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS, dev));
    std::string j = fmt("{\"attr_wait_value_nor\":%d,\"attr_stream_mem_ops_v1\":%d,\"attr_64bit_mem_ops\":%d", nor, v1, b64);
    if (!nor && !v1) return j + ",\"supported\":false,\"error\":\"driver reports no stream memory ops support\"}";
    CK(cudaFree(0)); // ensure primary context is current
    Stream sa, sb; DevBuf d(64), kb(NSMALL * 4); CK(cudaMemset(d.p, 0, 64)); CK(cudaMemset(kb.p, 0, NSMALL * 4));
    Event e; std::vector<double> lat;
    for (unsigned i = 1; i <= 60; i++) {
        CKD(cuStreamWaitValue32((CUstream)(cudaStream_t)sa, (CUdeviceptr)d.p, i, CU_STREAM_WAIT_VALUE_GEQ));
        k_small<<<40, 256, 0, sa>>>((float*)kb.p); CK(cudaEventRecord(e.e, sa));
        std::this_thread::sleep_for(std::chrono::microseconds(200)); // let the wait become pending
        double t0 = now_us(); k_store<<<1, 1, 0, sb>>>((volatile unsigned*)d.p, i); CK(cudaGetLastError());
        for (;;) { cudaError_t q = cudaEventQuery(e.e); if (q == cudaSuccess) break; if (q != cudaErrorNotReady) CK(q);
                   if (now_us() - t0 > 2e6) throw std::runtime_error("wait-value not satisfied within 2 s"); }
        if (i > 10) lat.push_back(now_us() - t0);
    }
    return j + ",\"supported\":true,\"other_stream_write_to_dependent_kernel_done_us\":" + st(lat) + "}";
}

// ---------- main ----------
int main() {
    std::string out = "{";
    try {
        CK(cudaSetDevice(0)); cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
        int dv = 0, rv = 0; cudaDriverGetVersion(&dv); cudaRuntimeGetVersion(&rv);
        out += fmt("\"device\":{\"name\":\"%s\",\"compute_capability\":\"%d.%d\",\"driver_version\":%d,\"runtime_version\":%d,\"total_memory_bytes\":%zu}",
                   p.name, p.major, p.minor, dv, rv, p.totalGlobalMem);
    } catch (std::exception& x) { out += "\"device\":" + jerr(x.what()); }
    struct { const char* key; std::string (*fn)(); } exps[] = {
        {"launch_floor", exp_A}, {"sync_cost", exp_B}, {"h2d_copies", exp_C}, {"worker_contention", exp_D},
        {"mapped_mailbox", exp_E}, {"wait_value", exp_F}};
    for (auto& e : exps) {
        fprintf(stderr, "[%s] start\n", e.key);
        std::string r;
        try { r = e.fn(); } catch (std::exception& x) { r = jerr(x.what()); }
        cudaGetLastError();
        cudaDeviceSynchronize();
        fprintf(stderr, "[%s] done\n", e.key);
        out += std::string(",\"") + e.key + "\":" + r;
    }
    out += "}";
    printf("%s\n", out.c_str());
    cudaDeviceReset();
    return 0;
}
