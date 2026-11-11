# SPDX-License-Identifier: Apache-2.0
# NarrativeQA doc-QA mooncake_trace (combined warmup + scored), copy of gen_nqa.py with a configurable working set.
# usage: gen_nqa_l1l2.py WS_TOKENS RATE DUR_S OUT.jsonl [PKL]
# Docs: NarrativeQA test docs (Step-3.7 token counts) that fit 262144, drawn in a seeded random order until WS_TOKENS doc tokens.
# Fit with 2% headroom: AIPerf's synthesized text tokenizes ~1.1% longer than input_length (2026-10-11: 259,230 -> 262,081).
# Past the pool (25.0M tokens), further passes walk fresh permutations of it; each copy is a new doc index, so it
# gets fresh hash_ids and AIPerf synthesizes distinct content per hash_id (Bo 2026-10-11: working set into L2).
# Pass 0 is the original selection, so WS <= pool gives the same trace as before.
# Warmup: one request per doc (question 0, OSL 1), paced at 20k prompt tokens/s (measured warmup compute rate).
# Scored: starts 30 s after the paced warmup ends; Poisson RATE req/s for DUR_S, uniform iid doc draws, OSL 64.
# 512-token hash_ids: doc blocks shared per doc, question tail blocks unique (as gen_nqa.py).
import json, pickle, sys
import numpy as np
SYS, B, OSL, WARM_TPS, GAP = 64, 512, 64, 20000.0, 30.0
WS, RATE, DUR, OUT = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
PKL = sys.argv[5] if len(sys.argv) > 5 else "/raid/bo/prt-cache/tmp_analysis/workload2/docqa_norm.pkl"
docs_all = [d for d in pickle.load(open(PKL, "rb"))["narrativeqa"] if (SYS + d[1] + max(d[2])) * 1.02 + 256 <= 262144]
rng = np.random.default_rng(1)
sel = []; ws = 0; passes = 0
while True:
    added = 0
    for i in rng.permutation(len(docs_all)):
        if ws + docs_all[i][1] > WS: continue
        sel.append(docs_all[i]); ws += docs_all[i][1]; added += 1
    passes += 1
    if added < len(docs_all): break   # pool not exhausted in this pass: WS reached
assert len(sel) * 2000 < 10**7, "doc hash_id range would collide with question tail ids"
nid = [10**7]
def row(t, di, k, osl):
    _, D, Q, _ = sel[di]; isl = SYS + D + Q[k % len(Q)]
    shared = (SYS + D) // B; nb = -(-isl // B)
    ids = [di * 2000 + b for b in range(shared)] + list(range(nid[0], nid[0] + nb - shared)); nid[0] += nb - shared
    return dict(timestamp=int(t * 1000), input_length=isl, output_length=osl, hash_ids=ids)
rows = []; c = 0
for di in range(len(sel)):
    r = row(c / WARM_TPS, di, 0, 1); rows.append(r); c += r["input_length"]
t0 = c / WARM_TPS + GAP; t = 0.0; cnt = {}; n = 0; isl_sum = 0
while True:
    t += rng.exponential(1 / RATE)
    if t > DUR: break
    di = int(rng.integers(len(sel))); k = cnt.get(di, 1); cnt[di] = k + 1
    r = row(t0 + t, di, k, OSL); rows.append(r); n += 1; isl_sum += r["input_length"]
with open(OUT, "x") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
print(json.dumps(dict(docs=len(sel), docs_available=len(docs_all), passes=passes, ws_doc_tokens=ws, warmup_rows=len(sel),
                      warmup_tokens=c, warmup_paced_s=c / WARM_TPS, scored_start_s=t0, scored_rows=n,
                      scored_mean_isl=isl_sum / max(n, 1), scored_prompt_tps=isl_sum / DUR, last_ts_s=rows[-1]["timestamp"] / 1000)))
