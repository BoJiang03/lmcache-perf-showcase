# Serving config

Decided 2026-10-07 (Bo): stock vLLM nightly with config-level tuning only; source-level vLLM
optimizations are kept out of this repo (Bo's own `prt-vllm-opt`, built FROM this serving image). LMCache, vLLM native
offloading and Mooncake run as official nightly, no instrumentation. Values below are the ones measured in the e2e
ramps (records 2026-10-03 to 10-05, on the patched stack); `config.toml` is the machine-readable form.

## vLLM (Step-3.7-Flash-FP8, TP4 on 4 x H200)

| setting | value | why |
|---|---|---|
| `VLLM_USE_V2_MODEL_RUNNER` | `1` | MTP accept length 2.47 -> 2.78; N 118 -> 120 |
| `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR` | `/raid/bo/prt-cache/vllm-cache/flashinfer_autotune_stock` | startup only: reuses per-rank autotune results, autotune 9 -> 2 s; tactics fixed across runs |
| `VLLM_WORKER_MULTIPROC_METHOD` | `fork` | startup only: spawn starts the 4 workers one after another (~17 s each) |
| `VLLM_HTTP_TIMEOUT_KEEP_ALIVE` | `600` | above AIPerf's 300 s pool keepalive; the 5 s default races with connection reuse (2 errors at N=130, requests never reached vLLM) |
| `--compilation-config` | `{"pass_config":{"fi_allreduce_fusion_max_size_mb":2}}` | the H200 TP4 default; set so the API server does not import FlashInfer, which initializes CUDA and forces spawn |
| `--tensor-parallel-size` / `--enable-expert-parallel` | `4` / on | without EP p10 22.7 vs ~30 |
| `--kv-cache-dtype` | `fp8` | 2x GPU KV capacity |
| `--speculative-config` | `{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic","rejection_sample_method":"block"}` | lossless; N=124 1/3 -> 3/3 passes |
| `--max-num-batched-tokens` | `8192` | 4096 / 16384 no better |
| `--gpu-memory-utilization` / `--max-num-seqs` | `0.95` / `256` | |
| `--enable-prefix-caching`, `--disable-cascade-attn` | on | Step recipe |
| model flags | `--trust-remote-code --language-model-only --generation-config vllm --reasoning-parser step3p5 --enable-auto-tool-choice --tool-call-parser step3p5 --enable-prompt-tokens-details` | |
| connector | `--kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"tcp://localhost","lmcache.mp.port":16555,"lmcache.mp.mq_timeout":1500}}'` | |

Left at default: attention backend (FA3), cudagraph mode and capture size (512), async scheduling.

## LMCache MP server

```
LMCACHE_LOG_LEVEL=WARNING lmcache server --host 127.0.0.1 --port 16555 --http-port 18080 \
  --chunk-size 1024 --separate-object-groups --disable-observability \
  --l1-size-gb 800 --eviction-policy LRU --eviction-trigger-watermark 0.95 --eviction-ratio 0.05 \
  --l2-prefetch-policy retain \
  --l2-adapter '{"type":"fs_native","base_path":"/raid/bo/prt-cache/l2","num_workers":16,"max_capacity_gb":1500,
                 "eviction":{"eviction_policy":"LRU","trigger_watermark":0.9,"eviction_ratio":0.1}}'
```

| setting | effect (measured) |
|---|---|
| chunk 1024, observability off, WARNING logs | boundary N 80 -> 104 |
| L1 watermark 0.95 / ratio 0.05, L2 + prefetch `retain` | N 104 -> 112 |
| L2 1.5 TB, L2 trigger 0.9 | N 112 -> 118 |
| L1 800 GiB | L1 ~2x L0 in bytes (L0 = 4 x 80 GiB) |

## Workload-specific

- agentic: N = 220 concurrent sessions, just under the stock KV knee (215 bounded, 230 diverges), AIPerf with the `weka_framing` and `live_history` plugins.
- nqa: `--kv-cache-memory-bytes 29334000000` (caps GPU KV at ~4M tokens so hits come from LMCache).

## Tried, not adopted

MBT 4096/16384, LPT, MTP 2, cudagraph capture 4096, FlashInfer/XQA attention, no EP; LMCache multi-server,
4+4 workers, eager prefetch, ARC eviction.
