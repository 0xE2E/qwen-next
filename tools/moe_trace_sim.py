"""MoE cache trace analysis + cache policy simulation (decode steps only). Usage: python moe_trace_sim.py trace.bin [n_req=3]"""
import sys, struct, heapq, collections
import numpy as np

def parse(path):
    b = open(path, "rb").read(); o = 0
    assert b[:4] == b"MCT1"; o = 4
    nl, nu, ng = struct.unpack_from("<3I", b, o); o += 12
    for _ in range(ng):
        _id, ne, nly = struct.unpack_from("<3I", b, o); o += 12 + 4 * nly
        o += 8 * 2 + 8 * 3
        nt, = struct.unpack_from("<I", b, o); o += 4 + 8 * nt
    recs = []
    n = len(b)
    while o + 12 <= n:
        il, used, nid = struct.unpack_from("<3i", b, o); o += 12
        if o + 4 * nid > n: break
        recs.append((il, used, np.frombuffer(b, dtype="<i4", count=nid, offset=o))); o += 4 * nid
    return nl, nu, recs

def main():
    path = sys.argv[1]; nreq = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    NL, NU, recs = parse(path)
    NE = 512
    sizes = collections.Counter(len(r[2]) // r[1] for r in recs)
    print(f"layers={NL} experts_used={NU} records={len(recs)}; tokens-per-record histogram (top): {sizes.most_common(8)}")
    # split into steps: a step is NL consecutive records for layers 0..NL-1 (records appear in layer order per batch)
    batches = []  # (ntok, list of per-layer arrays)
    cur = {}
    for il, used, ids in recs:
        if il in cur:  # new batch started
            batches.append(cur); cur = {}
        cur[il] = ids
    if cur: batches.append(cur)
    small = [bt for bt in batches if all(len(v) == NU for v in bt.values()) and len(bt) == NL]
    big = [bt for bt in batches if any(len(v) > NU for v in bt.values())]
    print(f"batches={len(batches)} decode(1-token, all {NL} layers)={len(small)} large-batch={len(big)} other={len(batches)-len(small)-len(big)}")
    if big: print("large-batch token counts:", sorted({len(next(iter(bt.values())))//NU for bt in big})[:10])
    D = np.array([[bt[l] for l in range(NL)] for bt in small])  # [steps, NL, NU]
    S = len(D)
    if S == 0: print("no decode steps"); return
    print(f"\n== A. decode steps={S} layers={NL} selections/step/layer={NU}")
    # request boundaries: the three requests are contiguous; the warmup (16 tokens) has few steps. Detect via large-batch records between decode batches.
    order = []  # decode step index at which each large-batch block occurred
    idx = 0; req_starts = []; prev_big = False; si = 0
    for bt in batches:
        isbig = any(len(v) > NU for v in bt.values())
        issmall = (not isbig) and len(bt) == NL
        if isbig: prev_big = True
        elif issmall:
            if prev_big: req_starts.append(si); prev_big = False
            si += 1
    if not req_starts or req_starts[0] != 0: req_starts = [0] + req_starts
    print("decode-step index where each request's decoding starts (after a large-batch prefill):", req_starts)
    # take the longest segments as the generation requests
    bounds = req_starts + [S]
    segs = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
    segs = [s for s in segs if s[1] - s[0] > 100]
    print("generation segments (steps):", [(a, b, b - a) for a, b in segs])
    a0, b0 = segs[-1] if segs else (0, S)
    G = D[a0:b0]
    key = (np.arange(NL)[None, :, None] * NE + G).astype(np.int64)
    flat = key.ravel()
    cnt = np.bincount(flat, minlength=NL * NE)
    print(f"distinct (layer,expert) pairs in one generation ({len(G)} steps): {(cnt>0).sum()} of {NL*NE}")
    cs = np.sort(cnt)[::-1]; tot = cs.sum()
    for p in (5, 10, 14, 20, 30):
        k = int(NL * NE * p / 100)
        print(f"  top {p:>2}% pairs ({k}) carry {100*cs[:k].sum()/tot:.1f}% of selections")
    # B temporal locality
    print("\n== B. temporal locality (all decode steps)")
    ov = np.zeros(NL)
    for l in range(NL):
        x = D[:, l, :]
        ov[l] = np.mean([len(set(x[t]) & set(x[t - 1])) for t in range(1, S)])
    print(f"overlap(t,t-1) out of {NU}: mean over layers {ov.mean():.2f}, min {ov.min():.2f} (L{ov.argmin()}), max {ov.max():.2f} (L{ov.argmax()})")
    for W in (4, 16, 64):
        fr = np.zeros(NL)
        for l in range(NL):
            x = D[:, l, :]; hit = 0; tot_ = 0
            for t in range(1, S):
                win = set(x[max(0, t - W):t].ravel().tolist())
                hit += sum(1 for e in x[t] if e in win); tot_ += NU
            fr[l] = hit / tot_
        print(f"fraction of selections seen in previous {W:>2} steps of the same layer: mean {fr.mean():.3f} min {fr.min():.3f} max {fr.max():.3f}")
    # C simulation
    Ks = (2000, 3424, 4500, 6000, 9000)
    def seqs(segs_):
        return np.concatenate([(np.arange(NL)[None, :, None] * NE + D[a:b]).reshape(-1, NU) for a, b in segs_]) if segs_ else None
    # accesses grouped by step: [steps*NL, NU] rows; each row = one lookup; entries of a row are "needed now"
    def lru(rows, K, prefill=None):
        od = collections.OrderedDict(); miss = 0; tot_ = 0; permiss = np.zeros(NL, dtype=np.int64)
        if prefill is not None:
            for k in prefill: od[int(k)] = 1
        for row in rows:
            row = row.tolist(); need = set(row)
            for k in row:
                tot_ += 1
                if k in od: od.move_to_end(k)
                else:
                    miss += 1; permiss[k // NE] += 1
                    if len(od) >= K:
                        for ek in od:
                            if ek not in need: del od[ek]; break
                    od[k] = 1
        return 1 - miss / tot_, permiss
    def lfu(rows, K, prefill=None, half=128):
        import math
        decay = math.log(2) / (half * NL)  # half-life in steps; rows advance NL per step
        sc = {}; last = {}; miss = 0; tot_ = 0; t = 0
        # lazy score: s*exp(-decay*(t-last)); evict min via full scan on eviction (sampled) -> use heap with approximate: scan all
        def cur(k): return sc[k] * math.exp(-decay * (t - last[k]))
        if prefill is not None:
            for k in prefill: sc[int(k)] = 1.0; last[int(k)] = 0
        for row in rows:
            row = row.tolist(); need = set(row)
            for k in row:
                tot_ += 1
                if k in sc: sc[k] = cur(k) + 1.0; last[k] = t
                else:
                    miss += 1
                    if len(sc) >= K:
                        # evict lowest score among sampled 64 entries (not needed now)
                        best = None; bv = 1e30
                        import random
                        cand = random.sample(list(sc.keys()), 64) if len(sc) > 64 else list(sc.keys())
                        for c in cand:
                            if c in need: continue
                            v = cur(c)
                            if v < bv: bv = v; best = c
                        if best is not None: del sc[best]
                    sc[k] = 1.0; last[k] = t
            t += 1
        return 1 - miss / tot_
    def belady(rows, K, prefill=None):
        flat = rows.ravel(); n = len(flat)
        nxt = np.empty(n, dtype=np.int64); lastpos = {}
        for i in range(n - 1, -1, -1):
            k = int(flat[i]); nxt[i] = lastpos.get(k, 1 << 60); lastpos[k] = i
        first = {}
        for i in range(n):
            first.setdefault(int(flat[i]), i)
        incache = {}  # key -> next use pos
        heap = []  # (-nextuse, key)
        if prefill is not None:
            for k in prefill: incache[int(k)] = first.get(int(k), 1 << 60); heapq.heappush(heap, (-incache[int(k)], int(k)))
        miss = 0; NUr = rows.shape[1]
        for r in range(len(rows)):
            base = r * NUr; need = set(flat[base:base + NUr].tolist())
            for j in range(NUr):
                i = base + j; k = int(flat[i])
                if k not in incache:
                    miss += 1
                    if len(incache) >= K:
                        skipped = []
                        while True:
                            nu, ek = heapq.heappop(heap)
                            if ek not in incache or incache[ek] != -nu: continue
                            if ek in need: skipped.append((nu, ek)); continue
                            del incache[ek]; break
                        for s in skipped: heapq.heappush(heap, s)
                incache[k] = int(nxt[i]); heapq.heappush(heap, (-int(nxt[i]), k))
        return 1 - miss / n
    allsegs = segs if segs else [(0, S)]
    print("\n== C. cache simulation (hit rate over decode lookups; K slots shared by all layers)")
    cold_rows = seqs([allsegs[-1]]); cont_rows = seqs(allsegs)
    print(f"cold: single request ({len(cold_rows)//NL} steps) starting empty; continuing: {len(allsegs)} requests back to back (identical repeated prompt flatters later requests)")
    hdr = f"{'K':>6} | " + " | ".join(f"{r+'/'+p:>14}" for r in ('cold', 'cont') for p in ('LRU', 'LFU', 'Belady'))
    print(hdr)
    for K in Ks:
        out = []
        for rows in (cold_rows, cont_rows):
            out += [lru(rows, K)[0], lfu(rows, K), belady(rows, K)]
        print(f"{K:>6} | " + " | ".join(f"{100*x:13.1f}%" for x in out), flush=True)
    # D prefill preload
    print("\n== D. preload with K most-used pairs of the PREFILL (large-batch) records of the same request")
    if not any(len(next(iter(bt.values())))//NU >= 64 for bt in big):
        print("trace contains no real prefill selections (only", len(big), "tiny batches of <=12 tokens; the bypass skips the cache for large batches) -> preload analysis not possible")
    else:
        # prefill of the last request: last large-batch block group before segs[-1]
        # collect big batches immediately preceding last generation: those after previous decode steps
        pc = np.zeros(NL * NE, dtype=np.int64); lastbig = []
        seen_small = 0; target = a0
        for bt in batches:
            isbig = any(len(v) > NU for v in bt.values())
            if isbig and seen_small == target: lastbig.append(bt)
            elif (not isbig) and len(bt) == NL: seen_small += 1
        for bt in lastbig:
            for l, ids in bt.items(): pc += np.bincount(l * NE + ids, minlength=NL * NE)
        print(f"prefill batches for the last request: {len(lastbig)}; distinct pairs {(pc>0).sum()}")
        order = np.argsort(-pc)
        for K in Ks:
            pre = order[:K]
            print(f"K={K}: cold+preload LRU {100*lru(cold_rows,K,pre)[0]:.1f}%  Belady {100*belady(cold_rows,K,pre)*1:.1f}%")
    print("\n== E. per-layer misses, LRU K=3424, continuing regime (misses per decode step per layer, last request)")
    hr, pm = lru(cont_rows, 3424)
    nsteps = len(cont_rows) // NL
    print(f"overall hit {100*hr:.1f}%, mean misses/step/layer {pm.sum()/nsteps/NL:.2f}, total/step {pm.sum()/nsteps:.1f}")
    top = np.argsort(-pm)[:10]
    print("worst layers (misses/step):", ", ".join(f"L{l}:{pm[l]/nsteps:.2f}" for l in top))
    print("best layers:", ", ".join(f"L{l}:{pm[l]/nsteps:.2f}" for l in np.argsort(pm)[:5]))
    q = [pm[i*12:(i+1)*12].sum()/nsteps/12 for i in range(4)]
    print("misses/step/layer by quarter of depth (L0-11,12-23,24-35,36-47):", [round(float(x), 2) for x in q])

main()
