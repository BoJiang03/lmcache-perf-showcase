# 2026-11-13: PRT alignment, case 3 (MiMo TP2), overnight reruns of cases 1-3

## Decisions (user)
- Align every deployment with PRT (LMCache PR #5744): LMCache `--max-workers` = ranks,
  `--l1-read-ttl-seconds 3600`, V2 runner + fork, `--disable-cascade-attn`, fusion
  compile config, MiniMax recipe; cache-bust `system_suffix`; keep chunk 256, spec on,
  GLM mbt 2048 / max-num-seqs 16.
- Case 3 on MiMo-V2.6-Flash-RL at TP2 (TP4's GPU KV pool was too large).
- Lazy SWA zero-KV bug: option 1 (keep chunk 256; mask chunks missing a kept window block,
  pin window tails in lazy offload).
- Case 3 sizing: conc 88 (GPU KV peak ~100%), L2 = GPU memory = 282 GB, L1 64 GB.
- Native arms may be rerun too; package a repro repo if gains are large and plausible;
  commit locally, push only on the user's word.

## Findings
- MiMo TP2 produced garbage under torch.compile (eager fine, fusion passes irrelevant);
  first seen as spec acceptance 1.00. Cause: padded fused-QKV `_split_qkv` view chain, only
  taken at TP2. Fixed with explicit slices + cat; DFlash acceptance 4.89. All earlier TP2
  pilot data was invalid.
- L2 per token: tuned lazy ~31 KB, eager baseline ~143 KB (SWA window tail for every chunk).
  LRU replay of the c48 pilot: 250 GB covers 99.8% (lazy) vs 90.7% (eager) of reuse.
- c48 GPU KV p50 0.45 / max 0.60, c64 0.58 / 0.78, roughly linear in conc.

## Results (960 s window)
- Case 3: tuned vs baseline 1185 vs 374 req, TTFT p50 2.7 vs 38.5 s, total tok/s 101.6K vs
  30.8K, recomputed 6.7% vs 46.1%.
- Case 1: LMCache vs native 621 vs 178 req, TTFT p50 18.1 vs 102.1 s, tok/s 46.4K vs 14.1K.
- Case 2: LMCache vs native 1535 vs 636 req, TTFT p50 29.6 vs 81.5 s, tok/s 127.5K vs 52.9K.
  PRT alignment halved LMCache TTFT (max-workers) and lifted native 277 -> 636 req.
- Full tables: /home/bo/lmcache-showcase-repro/RESULTS.md.

## Commits (local, not pushed)
- LMCache `lazy-final-gqadedup` 5b8d231a: no zeroed sliding-window tails under lazy offload.
- vLLM `mimo-tp2-split-qkv-inductor-fix` b49e18562 on 4e3463c87 ($PRT_W/src/vllm).
- Repro kit: /home/bo/lmcache-showcase-repro bf0595a.
- The patched vLLM file is also installed in $PRT_W/venv (original backed up).
