#!/usr/bin/env python3
"""Run qnext_vramlimit.exe while sampling Windows GPU memory counters and nvidia-smi.

Usage (from tools/s0, no llama-server or other GPU user running):
    python run_vramlimit.py --mode malloc
    python run_vramlimit.py --mode vmm
    python run_vramlimit.py --mode malloc --start-mib 14000 --step-mib 32 --max-mib 16150

Output: measurements/s0/vramlimit_<mode>.json (repository root) and a compact table on stdout.
Samplers (250 ms nvidia-smi poll, back-to-back Get-Counter loop in one persistent PowerShell) run for the whole
probe lifetime. The probe runs with --hold-ms 300 so each step line is followed by a 300 ms pause for sampling.
The probe is killed if it prints nothing for --step-timeout seconds (default 60).
"""
import argparse, json, os, re, subprocess, sys, threading, time, queue

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "..", "measurements", "s0")

PS_LOOP = r"""
$ErrorActionPreference = 'SilentlyContinue'
$paths = @('\GPU Adapter Memory(*)\Dedicated Usage', '\GPU Adapter Memory(*)\Shared Usage')
while ($true) {
  $c = Get-Counter -Counter $paths
  $s = ($c.CounterSamples | ForEach-Object { $_.Path + '=' + [int64]$_.CookedValue }) -join ';'
  [Console]::Out.WriteLine($s); [Console]::Out.Flush()
}
"""

def smi_query():
    """returns (used, reserved_or_None, free) in MiB or None."""
    for fields, has_res in (("memory.used,memory.reserved,memory.free", True), ("memory.used,memory.free", False)):
        try:
            r = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=10)
            if r.returncode != 0:
                continue
            v = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
            if has_res:
                return float(v[0]), float(v[1]), float(v[2])
            return float(v[0]), None, float(v[1])
        except Exception:
            continue
    return None

def smi_sampler(stop, out):
    while not stop.is_set():
        t = time.time()
        q = smi_query()
        if q:
            out.append({"t": t, "used": q[0], "reserved": q[1], "free": q[2]})
        dt = 0.25 - (time.time() - t)
        if dt > 0:
            stop.wait(dt)

PATH_RE = re.compile(r"\(([^)]*)\)\\(dedicated usage|shared usage)=(-?\d+)$", re.I)

def counter_sampler(stop, out, proc_holder):
    p = subprocess.Popen(["powershell", "-NoProfile", "-NonInteractive", "-Command", PS_LOOP],
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    proc_holder.append(p)
    for line in p.stdout:
        t = time.time()
        inst = {}
        for item in line.strip().split(";"):
            m = PATH_RE.search(item.strip())
            if m:
                d = inst.setdefault(m.group(1), {})
                d["ded" if m.group(2).lower().startswith("ded") else "shr"] = int(m.group(3)) / 1048576.0
        if inst:
            out.append({"t": t, "inst": inst})
        if stop.is_set():
            break

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["malloc", "vmm"], default="malloc")
    ap.add_argument("--start-mib", type=int, default=13000)
    ap.add_argument("--step-mib", type=int, default=64)
    ap.add_argument("--max-mib", type=int, default=16150)
    ap.add_argument("--slow-factor", type=float, default=3.0)
    ap.add_argument("--probe", default=os.path.join(HERE, "qnext_vramlimit.exe"))
    ap.add_argument("--step-timeout", type=float, default=60.0)
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    out_json = os.path.join(OUT_DIR, f"vramlimit_{a.mode}.json")
    probe_json = os.path.join(OUT_DIR, f"vramlimit_{a.mode}_probe.json")

    start_smi = smi_query()
    if not start_smi:
        print("nvidia-smi query failed; aborting"); return 1
    print(f"nvidia-smi at start: used={start_smi[0]} reserved={start_smi[1]} free={start_smi[2]} MiB")

    stop = threading.Event()
    smi, ctr, holder = [], [], []
    ts = threading.Thread(target=smi_sampler, args=(stop, smi), daemon=True)
    tc = threading.Thread(target=counter_sampler, args=(stop, ctr, holder), daemon=True)
    ts.start(); tc.start()
    t0 = time.time()
    while not ctr and time.time() - t0 < 20:
        time.sleep(0.2)
    if not ctr:
        print("WARNING: no Get-Counter samples after 20 s; continuing without OS counters")
    time.sleep(1.0)

    cmd = [a.probe, "--mode", a.mode, "--start-mib", str(a.start_mib), "--step-mib", str(a.step_mib),
           "--max-mib", str(a.max_mib), "--slow-factor", str(a.slow_factor), "--hold-ms", "300", "--out", probe_json]
    print("running:", " ".join(cmd))
    if os.path.exists(probe_json):
        os.remove(probe_json)
    probe = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    q = queue.Queue()
    def reader():
        for line in probe.stdout:
            q.put((time.time(), line.rstrip("\n")))
        q.put(None)
    threading.Thread(target=reader, daemon=True).start()

    lines, hung, last = [], False, time.time()
    try:
        while True:
            try:
                item = q.get(timeout=1.0)
            except queue.Empty:
                if time.time() - last > a.step_timeout:
                    hung = True
                    print(f"PROBE HUNG: no output for {a.step_timeout:.0f} s; killing")
                    probe.kill()
                    break
                continue
            if item is None:
                break
            last = item[0]
            lines.append(item)
            print("  probe>", item[1], flush=True)
    except KeyboardInterrupt:
        print("Ctrl-C: killing probe")
        probe.kill()
    try:
        probe.wait(timeout=30)
    except subprocess.TimeoutExpired:
        probe.kill(); probe.wait()
    time.sleep(1.0)
    stop.set()
    for p in holder:
        try: p.kill()
        except Exception: pass
    ts.join(timeout=5)

    # ---- confirm memory returned ----
    end_smi, deadline = None, time.time() + 15
    while time.time() < deadline:
        end_smi = smi_query()
        if end_smi and end_smi[0] == start_smi[0]:
            break
        time.sleep(1.0)
    restored = bool(end_smi and end_smi[0] == start_smi[0])

    # ---- identify NVIDIA instance: largest dedicated swing ----
    swing = {}
    for s in ctr:
        for k, v in s["inst"].items():
            if "ded" in v:
                lo, hi = swing.get(k, (1e18, -1e18))
                swing[k] = (min(lo, v["ded"]), max(hi, v["ded"]))
    nv = max(swing, key=lambda k: swing[k][1] - swing[k][0]) if swing else None

    def nearest(series, t):
        if not series: return None
        return min(series, key=lambda s: abs(s["t"] - t))

    steps = []
    for t, ln in lines:
        m = re.match(r"STEP (.*)", ln)
        if not m: continue
        kv = dict(x.split("=", 1) for x in m.group(1).split())
        tt = t + 0.15  # sampler window is the 300 ms hold after the line
        c, sm = nearest(ctr, tt), nearest(smi, tt)
        rec = {"t": t, **kv}
        if c and nv and nv in c["inst"]:
            rec["dedicated_mib"] = c["inst"][nv].get("ded"); rec["shared_mib"] = c["inst"][nv].get("shr")
            rec["counter_dt_s"] = c["t"] - tt
            rec["all_instances"] = c["inst"]
        if sm:
            rec["smi_used_mib"] = sm["used"]; rec["smi_reserved_mib"] = sm["reserved"]; rec["smi_free_mib"] = sm["free"]
            rec["smi_dt_s"] = sm["t"] - tt
        steps.append(rec)

    probe_result = None
    if os.path.exists(probe_json):
        try:
            with open(probe_json) as f: probe_result = json.load(f)
        except Exception as e:
            probe_result = {"error": str(e)}

    result = {"mode": a.mode, "args": vars(a), "probe_cmd": cmd, "probe_exit": probe.returncode, "probe_hung": hung,
              "nvidia_instance": nv, "instance_dedicated_swing_mib": {k: list(v) for k, v in swing.items()},
              "smi_start": start_smi, "smi_end": end_smi, "smi_restored_to_start": restored,
              "counter_samples": len(ctr), "smi_samples": len(smi),
              "probe_lines": [{"t": t, "line": ln} for t, ln in lines], "steps": steps, "probe_result": probe_result,
              "raw_counter_samples": ctr, "raw_smi_samples": smi}
    with open(out_json, "w") as f:
        json.dump(result, f, indent=1)

    # ---- table ----
    def g(r, k, w=8, f="{:.0f}"):
        v = r.get(k)
        try: return f.format(float(v)).rjust(w)
        except Exception: return "-".rjust(w)
    print()
    print(f"{'cum MiB':>8} {'read GB/s':>9} {'early GB/s':>10} {'cuda free':>9} {'dedic MiB':>9} {'shared MiB':>10} {'smi used':>9}")
    for r in steps:
        print(f"{g(r,'cum_mib')} {g(r,'read_gbps',9)} {g(r,'early_gbps',10)} {g(r,'cuda_free_mib',9)} "
              f"{g(r,'dedicated_mib',9)} {g(r,'shared_mib',10,'{:.0f}')} {g(r,'smi_used_mib',9)}   {r.get('rc','')}")
    print()
    for t, ln in lines:
        if ln.split(" ")[0] in ("START", "REF", "STOP", "INTEGRITY", "END", "INFO"):
            print(ln)
    print(f"NVIDIA counter instance: {nv}; samples: counters={len(ctr)} smi={len(smi)} over probe run")
    print(f"probe exit={probe.returncode} hung={hung}")
    print(f"nvidia-smi memory.used start={start_smi[0]} end={end_smi[0] if end_smi else None} -> "
          f"{'RESTORED' if restored else 'NOT RESTORED (check for stray GPU processes!)'}")
    print("wrote", out_json)
    return 0 if restored and not hung else 1

if __name__ == "__main__":
    sys.exit(main())
