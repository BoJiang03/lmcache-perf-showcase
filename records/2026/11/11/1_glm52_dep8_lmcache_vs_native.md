# GLM-5.2-FP8 DEP8: LMCache shared L1 vs vLLM native per-rank CPU offload

Date: 2026-11-11 (PT). Host: ali-us-h200-03 (8 x H200, 2 TiB RAM), shared with other users.

## Goal (Bo)

- Deploy GLM-5.2-FP8 with DP8 + EP8 based on the PRT config; compare vLLM native vs LMCache.
- vLLM native: OffloadingConnector, DEP8, each DP rank 150 GB CPU, no sharing across ranks.
- LMCache: one MP server, L1 1.2 TB, shared by all 8 ranks.
- Workload: agentic (AIPerf `inferencex-agentx-mvp`, weka cc-traces 256k), concurrency so that
  1.2 TB L1 is "barely enough".
- Routing: vLLM default (no prefix affinity). Warmup: AIPerf default spread (no `--burst-phase-starts`;
  Bo: spread is more realistic).
- Acceptance: native hit rate low, LMCache high; TTFT, total token throughput and interactivity
  (tokens/s/user) 2-3x better with LMCache.
- Point of the showcase (Bo): LMCache shares KV across DP ranks, vLLM does not. With load-based
  routing a session's turns spread over all ranks, so each native rank needs the whole working set:
  effective native capacity ~150 GiB, not 1.2 TB.

## Result (N=96, 960 s window, one run each)

| metric | vLLM native | LMCache | LMCache advantage |
|---|---|---|---|
| TTFT p50 (s) | 55.52 | 15.77 | 3.52x |
| TTFT p90 (s) | 421.42 | 157.90 | 2.67x |
| TTFT p95 (s) | 596.41 | 203.04 | 2.94x |
| TTFT mean (s) | 135.94 | 45.36 | 3.00x |
| Total tok/s | 13,908.67 | 47,359.19 | 3.41x |
| Input tok/s | 13,810.61 | 46,921.12 | 3.40x |
| Output tok/s | 98.06 | 438.07 | 4.47x |
| Interactivity mean (tok/s/user) | 10.15 | 31.16 | 3.07x |
| Interactivity p50 | 7.04 | 28.94 | 4.11x |
| Decode p10 | 4.49 | 17.30 | 3.85x |
| Completed requests | 189 | 620 | 3.28x |
| prompt-token hit rate (GPU+CPU) | 38.0% | 94.7% | |
| external (CPU tier) share | 23.7% | 83.8% | |

Open items:
- LMCache L1 peaked at 638 GiB with no eviction, so N=96 is not yet "barely enough" for 1.2 TB.
  Next: N ~150-180.
- One run per side, no repeats.

## Setup

- vLLM nightly 0.34.0rc1.dev156+g4e3463c87 (cu130); LMCache dev 29340c7c (built with gcc-toolset-12;
  host gcc 10 cannot build torch C++20 extensions); AIPerf 0.13.0 + weka_framing + live_history.
- No docker (bo was not in the docker group at the time): venvs under
  `/mnt/nvme0/yuhan/tmp/nvidia/bo-prt-glm52/` (bo had no writable dir on /mnt/nvme*).
- Driver: `glm_dep8/run_one.py` (reuses `runner/sampler.py` and `runner/score.py`).
- Trace dir digest matches config.toml (0766fc32...).
- Results: `/mnt/nvme0/yuhan/tmp/nvidia/bo-prt-glm52/results/{full_lmc_l1200_n96,full_native_150g_n96}`,
  table in `results/COMPARISON.md`.

Commands (both sides share every vLLM setting except the connector):

```
python3 -m glm_dep8.run_one --mode lmcache --l1-gb 1200 --concurrency 96 --duration 960 \
  --mbt 2048 --max-num-seqs 16 --gpu-util 0.94 --vllm-arg=--max-cudagraph-capture-size=32 \
  --out <W>/results/full_lmc_l1200_n96
python3 -m glm_dep8.run_one --mode native --concurrency 96 --duration 960 \
  --mbt 2048 --max-num-seqs 16 --gpu-util 0.94 --vllm-arg=--max-cudagraph-capture-size=32 \
  --out <W>/results/full_native_150g_n96
```

Native: `CPUOffloadingSpec`, `cpu_bytes_to_use` = 150 GiB per DP engine (checked in vLLM source:
DP engines have world_size 1, so the value is per rank, not split). LMCache: `--l1-size-gb 1200`,
chunk 256, LRU, watermark 0.95 / ratio 0.05, observability off, no L2. Native offload block 256.
No L2 / secondary tier on either side.

## Findings along the way

- Weights: 703.7 GiB total (experts 684.2, non-expert 19.6). DEP8 loads 111.1 GiB per GPU.
- KV per token (fp8_ds_mla + indexer): 53.5 KB. GPU KV per rank at util 0.94: ~295-307k tokens
  (1.13-1.17 x 256k).
- Activation peak dominated by MBT: at MBT 8192 KV memory was -5.4 GiB; 4096 -> 12.7 GiB (not
  enough for 256k); 2048 -> 19 GiB at util 0.975.
- Runtime OOM in FlashInfer CUTLASS fused-MoE workspace (`getWorkspaceInfo`, 1.4-2.9 GiB) at
  util 0.975 and 0.96 during long prefills; 0.94 is stable. The LMCache server also takes ~670 MiB
  CUDA context on each GPU, outside vLLM's budget.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is rejected with LMCacheMPConnector unless
  `--enable-cumem-allocator`; not used.
- vLLM native CPU offload always uses a `/dev/shm` mmap region on CUDA (8 x 161 GB); the default
  1008G tmpfs made native workers die at init. Fixed with
  `sudo mount -o remount,size=1800G /dev/shm` (until reboot).
- vLLM default DP routing (`core_client.py` get_core_engine_for_request): least loaded by
  max(waiting+running, in-flight), KV-pressure penalty on waiting, rotating tie-break (= round robin
  at equal load); no prefix/session affinity.
- AIPerf agentic warmup: 89 priming requests spread to align on t* (~40 min); the first-run
  FlashInfer MoE JIT takes ~15 min, cached afterwards.
- Shared-host contention: roy (object storage, ~1 TiB RAM; ghostServe GPU holder + ncu campaign),
  samuel (lmcache/vLLM, stale 280 GB /dev/shm file), yihua. Freed with Bo's sudo.
