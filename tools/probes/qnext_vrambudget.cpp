// qnext_vrambudget: account for the gap between physical VRAM, CUDA's free figure and
// nvidia-smi's figures, layer by layer (DXGI, D3DKMT, NVML, CUDA driver API).
// Phases: before_cuda -> cuda_context_active -> after_context_release.
// Only allocation: the CUDA primary context. All handles released on every path.
//
// Build (from an x64 VS developer prompt, i.e. after vcvars64.bat):
//   cl /nologo /O2 /EHsc /W3 qnext_vrambudget.cpp /Fe:qnext_vrambudget.exe /link dxgi.lib gdi32.lib
// Usage: qnext_vrambudget.exe [--out path.json]
#define NOMINMAX
#include <windows.h>
#include <winternl.h>
#define WIN32_NO_STATUS
#include <ntstatus.h>
#include <d3dkmthk.h>
#include <dxgi1_6.h>
#include <cstdarg>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

typedef unsigned long long u64;
static const double MiB = 1024.0 * 1024.0;
static std::string J;  // JSON accumulator
static char tb[2048];
static void jf(const char* fmt, ...) {
    va_list a; va_start(a, fmt); vsnprintf(tb, sizeof tb, fmt, a); va_end(a); J += tb;
}
static std::string esc(const std::string& s) {
    std::string o; for (char c : s) { if (c == '"' || c == '\\') o += '\\'; if ((unsigned char)c >= 32) o += c; } return o;
}
static std::string narrow(const wchar_t* w) {
    char b[512]; int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, b, sizeof b, 0, 0); return n > 0 ? std::string(b) : std::string();
}
static void say(const char* fmt, ...) { va_list a; va_start(a, fmt); vprintf(fmt, a); va_end(a); }

// Key figures captured per phase for the summary (MiB; -1 = unavailable)
struct Snap {
    double dxgi_nv_dedicated = -1, dxgi_local_budget = -1, dxgi_local_usage = -1, dxgi_local_avail_res = -1, dxgi_local_cur_res = -1;
    double kmt_segsize_dedicated = -1, kmt_group_local = -1, kmt_group_nonbudget = -1, kmt_seg_commitlimit = -1, kmt_seg_resident = -1;
    double nvml_total = -1, nvml_reserved = -1, nvml_free = -1, nvml_used = -1, bar1_total = -1;
};
static LUID g_nvLuid; static bool g_haveNv = false;

// ---------------- DXGI ----------------
static void phase_dxgi(Snap& s) {
    jf("\"dxgi\":{");
    IDXGIFactory4* f = nullptr;
    HRESULT hr = CreateDXGIFactory1(__uuidof(IDXGIFactory4), (void**)&f);
    if (FAILED(hr)) { jf("\"error\":\"CreateDXGIFactory1 hr=0x%08lx\"}", (unsigned long)hr); say("  DXGI: factory failed 0x%08lx\n", (unsigned long)hr); return; }
    jf("\"adapters\":[");
    for (UINT i = 0;; ++i) {
        IDXGIAdapter1* a1 = nullptr;
        if (f->EnumAdapters1(i, &a1) == DXGI_ERROR_NOT_FOUND) break;
        DXGI_ADAPTER_DESC1 d; a1->GetDesc1(&d);
        bool nv = d.VendorId == 0x10DE;
        if (nv && !g_haveNv) { g_nvLuid = d.AdapterLuid; g_haveNv = true; }
        say("  DXGI adapter %u: %s vendor=0x%04x LUID=%08lx-%08lx\n    Dedicated VRAM %.1f MiB, Dedicated SysMem %.1f MiB, Shared SysMem %.1f MiB flags=0x%x\n",
            i, narrow(d.Description).c_str(), d.VendorId, (unsigned long)d.AdapterLuid.HighPart, (unsigned long)d.AdapterLuid.LowPart,
            d.DedicatedVideoMemory / MiB, d.DedicatedSystemMemory / MiB, d.SharedSystemMemory / MiB, d.Flags);
        if (i) jf(",");
        jf("{\"index\":%u,\"description\":\"%s\",\"vendor_id\":%u,\"luid\":\"%08lx-%08lx\",\"flags\":%u,"
           "\"DedicatedVideoMemory\":%llu,\"DedicatedSystemMemory\":%llu,\"SharedSystemMemory\":%llu",
           i, esc(narrow(d.Description)).c_str(), d.VendorId, (unsigned long)d.AdapterLuid.HighPart, (unsigned long)d.AdapterLuid.LowPart, d.Flags,
           (u64)d.DedicatedVideoMemory, (u64)d.DedicatedSystemMemory, (u64)d.SharedSystemMemory);
        if (nv && s.dxgi_nv_dedicated < 0) s.dxgi_nv_dedicated = d.DedicatedVideoMemory / MiB;
        IDXGIAdapter3* a3 = nullptr;
        if (SUCCEEDED(a1->QueryInterface(__uuidof(IDXGIAdapter3), (void**)&a3))) {
            const char* names[2] = { "local", "non_local" };
            DXGI_MEMORY_SEGMENT_GROUP g[2] = { DXGI_MEMORY_SEGMENT_GROUP_LOCAL, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL };
            for (int k = 0; k < 2; ++k) {
                DXGI_QUERY_VIDEO_MEMORY_INFO m = {};
                HRESULT h2 = a3->QueryVideoMemoryInfo(0, g[k], &m);
                if (FAILED(h2)) { jf(",\"%s\":{\"error\":\"hr=0x%08lx\"}", names[k], (unsigned long)h2); continue; }
                say("    %-9s Budget %.1f  CurrentUsage %.1f  AvailableForReservation %.1f  CurrentReservation %.1f MiB\n", names[k],
                    m.Budget / MiB, m.CurrentUsage / MiB, m.AvailableForReservation / MiB, m.CurrentReservation / MiB);
                jf(",\"%s\":{\"Budget\":%llu,\"CurrentUsage\":%llu,\"AvailableForReservation\":%llu,\"CurrentReservation\":%llu}",
                   names[k], (u64)m.Budget, (u64)m.CurrentUsage, (u64)m.AvailableForReservation, (u64)m.CurrentReservation);
                if (nv && k == 0 && s.dxgi_local_budget < 0) {
                    s.dxgi_local_budget = m.Budget / MiB; s.dxgi_local_usage = m.CurrentUsage / MiB;
                    s.dxgi_local_avail_res = m.AvailableForReservation / MiB; s.dxgi_local_cur_res = m.CurrentReservation / MiB;
                }
            }
            a3->Release();
        }
        jf("}");
        a1->Release();
    }
    jf("]}");
    f->Release();
}

// ---------------- D3DKMT ----------------
static void phase_kmt(Snap& s) {
    jf("\"d3dkmt\":{");
    if (!g_haveNv) { jf("\"error\":\"no NVIDIA adapter found via DXGI\"}"); say("  D3DKMT: no NVIDIA adapter\n"); return; }
    D3DKMT_OPENADAPTERFROMLUID oa = {}; oa.AdapterLuid = g_nvLuid;
    NTSTATUS st = D3DKMTOpenAdapterFromLuid(&oa);
    if (st != 0) { jf("\"error\":\"D3DKMTOpenAdapterFromLuid status=0x%08lx\"}", (unsigned long)st); say("  D3DKMT: open failed 0x%08lx\n", (unsigned long)st); return; }
    // legacy segment size
    D3DKMT_SEGMENTSIZEINFO ss = {}; D3DKMT_QUERYADAPTERINFO qa = {};
    qa.hAdapter = oa.hAdapter; qa.Type = KMTQAITYPE_GETSEGMENTSIZE; qa.pPrivateDriverData = &ss; qa.PrivateDriverDataSize = sizeof ss;
    st = D3DKMTQueryAdapterInfo(&qa);
    if (st == 0) {
        say("  D3DKMT GETSEGMENTSIZE: DedicatedVideo %.1f  DedicatedSystem %.1f  SharedSystem %.1f MiB\n",
            ss.DedicatedVideoMemorySize / MiB, ss.DedicatedSystemMemorySize / MiB, ss.SharedSystemMemorySize / MiB);
        jf("\"segment_size\":{\"DedicatedVideoMemorySize\":%llu,\"DedicatedSystemMemorySize\":%llu,\"SharedSystemMemorySize\":%llu}",
           (u64)ss.DedicatedVideoMemorySize, (u64)ss.DedicatedSystemMemorySize, (u64)ss.SharedSystemMemorySize);
        s.kmt_segsize_dedicated = ss.DedicatedVideoMemorySize / MiB;
    } else { jf("\"segment_size\":{\"error\":\"status=0x%08lx\"}", (unsigned long)st); say("  D3DKMT GETSEGMENTSIZE failed 0x%08lx\n", (unsigned long)st); }
    // segment group size (WDDM 2.x; has NonBudgetMemory)
    D3DKMT_SEGMENTGROUPSIZEINFO gs = {}; qa = {};
    qa.hAdapter = oa.hAdapter; qa.Type = KMTQAITYPE_GETSEGMENTGROUPSIZE; qa.pPrivateDriverData = &gs; qa.PrivateDriverDataSize = sizeof gs;
    st = D3DKMTQueryAdapterInfo(&qa);
    if (st == 0) {
        say("  D3DKMT GETSEGMENTGROUPSIZE: Local %.1f  NonLocal %.1f  NonBudget %.1f MiB (legacy dedicated %.1f)\n",
            gs.LocalMemory / MiB, gs.NonLocalMemory / MiB, gs.NonBudgetMemory / MiB, gs.LegacyInfo.DedicatedVideoMemorySize / MiB);
        jf(",\"segment_group_size\":{\"LocalMemory\":%llu,\"NonLocalMemory\":%llu,\"NonBudgetMemory\":%llu,\"LegacyDedicatedVideoMemorySize\":%llu}",
           (u64)gs.LocalMemory, (u64)gs.NonLocalMemory, (u64)gs.NonBudgetMemory, (u64)gs.LegacyInfo.DedicatedVideoMemorySize);
        s.kmt_group_local = gs.LocalMemory / MiB; s.kmt_group_nonbudget = gs.NonBudgetMemory / MiB;
    } else { jf(",\"segment_group_size\":{\"error\":\"status=0x%08lx\"}", (unsigned long)st); say("  D3DKMT GETSEGMENTGROUPSIZE failed 0x%08lx\n", (unsigned long)st); }
    // statistics: adapter -> NbSegments, then each segment
    D3DKMT_CLOSEADAPTER ca = {}; ca.hAdapter = oa.hAdapter;
    D3DKMT_QUERYSTATISTICS q = {}; q.Type = D3DKMT_QUERYSTATISTICS_ADAPTER; q.AdapterLuid = g_nvLuid;
    st = D3DKMTQueryStatistics(&q);
    if (st != 0) { jf(",\"statistics\":{\"error\":\"ADAPTER status=0x%08lx\"}}", (unsigned long)st); say("  D3DKMT stats ADAPTER failed 0x%08lx\n", (unsigned long)st); D3DKMTCloseAdapter(&ca); return; }
    ULONG nseg = q.QueryResult.AdapterInformation.NbSegments;
    say("  D3DKMT NbSegments=%lu\n", nseg);
    jf(",\"statistics\":{\"NbSegments\":%lu,\"segments\":[", nseg);
    for (ULONG i = 0; i < nseg; ++i) {
        D3DKMT_QUERYSTATISTICS qs = {}; qs.Type = D3DKMT_QUERYSTATISTICS_SEGMENT; qs.AdapterLuid = g_nvLuid; qs.QuerySegment.SegmentId = i;
        st = D3DKMTQueryStatistics(&qs);
        if (i) jf(",");
        if (st != 0) { jf("{\"segment\":%lu,\"error\":\"status=0x%08lx\"}", i, (unsigned long)st); say("    seg %lu: failed 0x%08lx\n", i, (unsigned long)st); continue; }
        const D3DKMT_QUERYSTATISTICS_SEGMENT_INFORMATION& si = qs.QueryResult.SegmentInformation;
        say("    seg %lu: Aperture=%lu CommitLimit %.1f  BytesCommitted %.1f  BytesResident %.1f MiB  AllocsCommitted %lu AllocsResident %lu TotalBytesEvicted %.1f MiB SysMem=%d SegType=%d\n",
            i, si.Aperture, si.CommitLimit / MiB, si.BytesCommitted / MiB, si.BytesResident / MiB,
            si.Memory.AllocsCommitted, si.Memory.AllocsResident, si.Memory.TotalBytesEvicted / MiB,
            (int)si.SegmentProperties.SystemMemory, (int)si.SegmentProperties.SegmentType);
        jf("{\"segment\":%lu,\"Aperture\":%lu,\"CommitLimit\":%llu,\"BytesCommitted\":%llu,\"BytesResident\":%llu,"
           "\"AllocsCommitted\":%lu,\"AllocsResident\":%lu,\"TotalBytesEvicted\":%llu,\"SystemMemory\":%d,\"SegmentType\":%d,\"PopulatedByReservedDDRByFirmware\":%d}",
           i, si.Aperture, (u64)si.CommitLimit, (u64)si.BytesCommitted, (u64)si.BytesResident,
           si.Memory.AllocsCommitted, si.Memory.AllocsResident, (u64)si.Memory.TotalBytesEvicted,
           (int)si.SegmentProperties.SystemMemory, (int)si.SegmentProperties.SegmentType, (int)si.SegmentProperties.PopulatedByReservedDDRByFirmware);
        if (!si.Aperture && s.kmt_seg_commitlimit < 0) { s.kmt_seg_commitlimit = si.CommitLimit / MiB; s.kmt_seg_resident = si.BytesResident / MiB; }
    }
    jf("]}}");
    D3DKMTCloseAdapter(&ca);
}

// ---------------- NVML ----------------
struct nvmlMemory_v2_t { unsigned version; unsigned long long total, reserved, free, used; };
struct nvmlMemory_t { unsigned long long total, free, used; };
struct nvmlBAR1_t { unsigned long long bar1Total, bar1Free, bar1Used; };
typedef int (*pInit)(); typedef int (*pShut)();
typedef int (*pGetHandle)(unsigned, void**);
typedef int (*pMemV2)(void*, nvmlMemory_v2_t*);
typedef int (*pMem)(void*, nvmlMemory_t*);
typedef int (*pBar1)(void*, nvmlBAR1_t*);
static HMODULE g_nvml = nullptr;
static bool nvml_load() {
    const char* paths[] = { "C:\\Windows\\System32\\nvml.dll", "C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvml.dll", "nvml.dll" };
    for (auto p : paths) { g_nvml = LoadLibraryA(p); if (g_nvml) { say("NVML loaded from %s\n", p); return true; } }
    return false;
}
static void phase_nvml(Snap& s) {
    jf("\"nvml\":{");
    if (!g_nvml) { jf("\"error\":\"nvml.dll not available\"}"); say("  NVML: not available\n"); return; }
    pInit init = (pInit)GetProcAddress(g_nvml, "nvmlInit_v2"); pShut shut = (pShut)GetProcAddress(g_nvml, "nvmlShutdown");
    pGetHandle gh = (pGetHandle)GetProcAddress(g_nvml, "nvmlDeviceGetHandleByIndex_v2");
    pMemV2 m2 = (pMemV2)GetProcAddress(g_nvml, "nvmlDeviceGetMemoryInfo_v2");
    pMem m1 = (pMem)GetProcAddress(g_nvml, "nvmlDeviceGetMemoryInfo");
    pBar1 b1 = (pBar1)GetProcAddress(g_nvml, "nvmlDeviceGetBAR1MemoryInfo");
    int r = init ? init() : -1;
    if (r != 0) { jf("\"error\":\"nvmlInit rc=%d\"}", r); say("  NVML init failed rc=%d\n", r); return; }
    void* dev = nullptr; r = gh ? gh(0, &dev) : -1;
    if (r != 0) { jf("\"error\":\"GetHandleByIndex rc=%d\"}", r); say("  NVML handle failed rc=%d\n", r); if (shut) shut(); return; }
    bool first = true;
    if (m2) {
        nvmlMemory_v2_t m = {}; m.version = (unsigned)(sizeof(nvmlMemory_v2_t) | (2u << 24));
        r = m2(dev, &m);
        if (r == 0) {
            say("  NVML MemoryInfo_v2: total %.1f  reserved %.1f  free %.1f  used %.1f MiB\n", m.total / MiB, m.reserved / MiB, m.free / MiB, m.used / MiB);
            jf("\"memory_v2\":{\"total\":%llu,\"reserved\":%llu,\"free\":%llu,\"used\":%llu}", m.total, m.reserved, m.free, m.used); first = false;
            s.nvml_total = m.total / MiB; s.nvml_reserved = m.reserved / MiB; s.nvml_free = m.free / MiB; s.nvml_used = m.used / MiB;
        } else { jf("\"memory_v2\":{\"error\":\"rc=%d\"}", r); say("  NVML MemoryInfo_v2 rc=%d\n", r); first = false; }
    }
    if (s.nvml_total < 0 && m1) {
        nvmlMemory_t m = {};
        if (m1(dev, &m) == 0) {
            say("  NVML MemoryInfo(v1): total %.1f  free %.1f  used %.1f MiB\n", m.total / MiB, m.free / MiB, m.used / MiB);
            jf("%s\"memory_v1\":{\"total\":%llu,\"free\":%llu,\"used\":%llu}", first ? "" : ",", m.total, m.free, m.used); first = false;
            s.nvml_total = m.total / MiB; s.nvml_free = m.free / MiB; s.nvml_used = m.used / MiB;
        }
    }
    if (b1) {
        nvmlBAR1_t b = {};
        r = b1(dev, &b);
        if (r == 0) {
            say("  NVML BAR1: total %.1f  free %.1f  used %.1f MiB\n", b.bar1Total / MiB, b.bar1Free / MiB, b.bar1Used / MiB);
            jf("%s\"bar1\":{\"total\":%llu,\"free\":%llu,\"used\":%llu}", first ? "" : ",", b.bar1Total, b.bar1Free, b.bar1Used);
            s.bar1_total = b.bar1Total / MiB;
        } else jf("%s\"bar1\":{\"error\":\"rc=%d\"}", first ? "" : ",", r);
    }
    jf("}");
    if (shut) shut();
}

static Snap phase(const char* name, bool last = false) {
    say("\n=== PHASE: %s ===\n", name);
    Snap s;
    jf("\"%s\":{", name);
    phase_dxgi(s); jf(",");
    phase_kmt(s); jf(",");
    phase_nvml(s);
    jf("}%s", last ? "" : ",");
    return s;
}

// ---------------- CUDA driver (dynamic) ----------------
typedef int CUresult; typedef int CUdevice; typedef void* CUcontext;
typedef CUresult(__stdcall* pcuInit)(unsigned);
typedef CUresult(__stdcall* pcuDeviceGet)(CUdevice*, int);
typedef CUresult(__stdcall* pcuDeviceTotalMem)(size_t*, CUdevice);
typedef CUresult(__stdcall* pcuRetain)(CUcontext*, CUdevice);
typedef CUresult(__stdcall* pcuRelease)(CUdevice);
typedef CUresult(__stdcall* pcuSetCur)(CUcontext);
typedef CUresult(__stdcall* pcuMemGetInfo)(size_t*, size_t*);
typedef CUresult(__stdcall* pcuGetName)(char*, int, CUdevice);

static void row(const char* label, double v, double ref, const char* refname) {
    if (v < 0) { say("  %-52s %12s\n", label, "n/a"); return; }
    if (ref >= 0) say("  %-52s %12.1f   (%+.1f vs %s)\n", label, v, v - ref, refname);
    else say("  %-52s %12.1f\n", label, v);
}

int main(int argc, char** argv) {
    std::string outPath;
    for (int i = 1; i < argc; ++i) if (!strcmp(argv[i], "--out") && i + 1 < argc) outPath = argv[++i];
    const double PHYS = 16384.0;
    jf("{\"tool\":\"qnext_vrambudget\",\"units\":\"bytes\",\"physical_mib\":%.0f,", PHYS);
    nvml_load();
    HMODULE cu = LoadLibraryA("nvcuda.dll");
    Snap s0 = phase("before_cuda");   // DXGI/D3DKMT/NVML before any CUDA call

    double cu_total_dev = -1, cu_free = -1, cu_total_ctx = -1;
    Snap s1, s2; bool haveS1 = false;
    CUcontext ctx = nullptr; CUdevice dev = 0;
    jf("\"cuda\":{");
    if (!cu) { jf("\"error\":\"nvcuda.dll not loadable\"},"); say("\nCUDA: nvcuda.dll not loadable\n"); }
    else {
        auto cuInit = (pcuInit)GetProcAddress(cu, "cuInit");
        auto cuDeviceGet = (pcuDeviceGet)GetProcAddress(cu, "cuDeviceGet");
        auto cuTotal = (pcuDeviceTotalMem)GetProcAddress(cu, "cuDeviceTotalMem_v2");
        auto cuRetain = (pcuRetain)GetProcAddress(cu, "cuDevicePrimaryCtxRetain");
        auto cuRel = (pcuRelease)GetProcAddress(cu, "cuDevicePrimaryCtxRelease_v2");
        auto cuSet = (pcuSetCur)GetProcAddress(cu, "cuCtxSetCurrent");
        auto cuMem = (pcuMemGetInfo)GetProcAddress(cu, "cuMemGetInfo_v2");
        auto cuName = (pcuGetName)GetProcAddress(cu, "cuDeviceGetName");
        CUresult r = (cuInit && cuDeviceGet && cuTotal && cuRetain && cuRel && cuSet && cuMem) ? cuInit(0) : -1;
        if (r != 0) { jf("\"error\":\"missing symbols or cuInit rc=%d\"},", r); say("\nCUDA: init failed rc=%d\n", r); }
        else {
            r = cuDeviceGet(&dev, 0);
            char nm[256] = "?"; if (cuName) cuName(nm, sizeof nm, dev);
            size_t tot = 0; if (r == 0 && cuTotal(&tot, dev) == 0) cu_total_dev = tot / MiB;
            say("\nCUDA device 0 (%s): cuDeviceTotalMem %.1f MiB (before context)\n", nm, cu_total_dev);
            jf("\"device_name\":\"%s\",\"cuDeviceTotalMem\":%llu,", esc(nm).c_str(), (u64)tot);
            if (r == 0 && cuRetain(&ctx, dev) == 0) {
                if (cuSet(ctx) == 0) {
                    size_t fr = 0, tt = 0;
                    if (cuMem(&fr, &tt) == 0) { cu_free = fr / MiB; cu_total_ctx = tt / MiB; }
                    say("CUDA cuMemGetInfo with primary context: free %.1f MiB, total %.1f MiB\n", cu_free, cu_total_ctx);
                    jf("\"cuMemGetInfo\":{\"free\":%llu,\"total\":%llu},", (u64)fr, (u64)tt);
                }
                jf("\"note\":\"context active during next phase\"},");
                s1 = phase("cuda_context_active"); haveS1 = true;
                cuSet(nullptr);
                cuRel(dev); ctx = nullptr;
            } else jf("\"error\":\"primary ctx retain failed\"},");
        }
    }
    s2 = phase("after_context_release", true);
    jf("}");
    if (cu) FreeLibrary(cu);
    if (g_nvml) FreeLibrary(g_nvml);

    // ---------------- summary ----------------
    say("\n=== SUMMARY (MiB; physical = %.0f) ===\n", PHYS);
    row("Physical (assumed 16 GiB)", PHYS, -1, "");
    row("DXGI NVIDIA DedicatedVideoMemory", s0.dxgi_nv_dedicated, PHYS, "physical");
    row("D3DKMT GETSEGMENTSIZE DedicatedVideoMemorySize", s0.kmt_segsize_dedicated, PHYS, "physical");
    row("D3DKMT GETSEGMENTGROUPSIZE LocalMemory", s0.kmt_group_local, PHYS, "physical");
    row("D3DKMT GETSEGMENTGROUPSIZE NonBudgetMemory", s0.kmt_group_nonbudget, -1, "");
    row("D3DKMT local segment CommitLimit", s0.kmt_seg_commitlimit, PHYS, "physical");
    row("CUDA cuDeviceTotalMem", cu_total_dev, PHYS, "physical");
    row("CUDA cuMemGetInfo total (ctx)", cu_total_ctx, PHYS, "physical");
    row("CUDA cuMemGetInfo free (ctx)", cu_free, cu_total_ctx, "CUDA total");
    row("NVML total (before CUDA)", s0.nvml_total, PHYS, "physical");
    row("NVML reserved (before CUDA)", s0.nvml_reserved, -1, "");
    row("NVML free (before CUDA)", s0.nvml_free, s0.nvml_total, "NVML total");
    row("NVML used (before CUDA)", s0.nvml_used, -1, "");
    row("DXGI local Budget (before CUDA)", s0.dxgi_local_budget, s0.dxgi_nv_dedicated, "DXGI dedicated");
    row("DXGI local AvailableForReservation (before CUDA)", s0.dxgi_local_avail_res, -1, "");
    row("DXGI local CurrentUsage (before CUDA)", s0.dxgi_local_usage, -1, "");
    if (haveS1) {
        auto d = [&](const char* l, double a, double b) { if (a >= 0 && b >= 0) say("  %-52s %+12.1f\n", l, a - b); else say("  %-52s %12s\n", l, "n/a"); };
        say("  -- context cost (active minus before) --\n");
        d("DXGI local CurrentUsage", s1.dxgi_local_usage, s0.dxgi_local_usage);
        d("DXGI local Budget", s1.dxgi_local_budget, s0.dxgi_local_budget);
        d("DXGI local AvailableForReservation", s1.dxgi_local_avail_res, s0.dxgi_local_avail_res);
        d("D3DKMT local segment BytesResident", s1.kmt_seg_resident, s0.kmt_seg_resident);
        d("NVML used", s1.nvml_used, s0.nvml_used);
        d("NVML free", s1.nvml_free, s0.nvml_free);
        d("NVML reserved", s1.nvml_reserved, s0.nvml_reserved);
        say("  -- after release minus before --\n");
        d("DXGI local CurrentUsage", s2.dxgi_local_usage, s0.dxgi_local_usage);
        d("D3DKMT local segment BytesResident", s2.kmt_seg_resident, s0.kmt_seg_resident);
        d("NVML used", s2.nvml_used, s0.nvml_used);
        d("NVML free", s2.nvml_free, s0.nvml_free);
    }
    if (cu_free >= 0 && s0.nvml_free >= 0) {
        say("  -- the gap --\n");
        say("  %-52s %+12.1f\n", "NVML free(before CUDA) - CUDA free(ctx active)", s0.nvml_free - cu_free);
        say("  %-52s %+12.1f\n", "Physical - CUDA free(ctx active)", PHYS - cu_free);
        say("  %-52s %+12.1f\n", "CUDA total - CUDA free (ctx active)", cu_total_ctx - cu_free);
    }

    if (!outPath.empty()) {
        FILE* f = fopen(outPath.c_str(), "wb");
        if (f) { fwrite(J.data(), 1, J.size(), f); fclose(f); say("\nJSON written to %s\n", outPath.c_str()); }
        else say("\ncould not open %s for writing\n", outPath.c_str());
    }
    return 0;
}
