# SPDX-License-Identifier: Apache-2.0
"""One GLM-5.2-FP8 DEP8 run without docker: vLLM native offload vs LMCache.

    python3 -m glm_dep8.run_one --mode lmcache|native --concurrency N --out DIR
        [--duration 960] [--max-model-len 202752] [--vllm-arg --flag=value ...]

Both modes share every vLLM setting (DP8 + EP8, default DP routing, i.e. no
prefix affinity); they differ only in the KV connector:

* native:  OffloadingConnector, CPUOffloadingSpec, 150 GB of CPU memory per DP
           rank (8 private CPU caches, nothing shared between ranks).
* lmcache: LMCacheMPConnector -> one LMCache MP server, L1 1.2 TB shared by all
           8 ranks. No L2 on either side.

Output in DIR: vllm.log, lmcache.log, aiperf.log, aiperf/, samples.jsonl,
meta.json, result.json (runner.score metrics + throughput/interactivity).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from runner.sampler import Sampler  # noqa: E402
from runner.score import load_requests, pct, score, window_bounds  # noqa: E402

W = Path(os.environ.get("PRT_W", "/mnt/nvme0/yuhan/tmp/nvidia/bo-prt-glm52"))
MODEL = W / "models" / "GLM-5.2-FP8"
HF_REPO = "zai-org/GLM-5.2-FP8"
SERVED = "glm52"
VLLM_PORT = 18000
LMC_PORT, LMC_HTTP = 16555, 18080
CPU_PER_RANK = 150 * 2**30  # 150 GiB per DP rank; 8 x 150 = 1200 GiB = LMCache L1
L1_GB = 1200  # LMCache GiB-based flag; see meta for bytes
DP = 8

VLLM_ENV = {
    "VLLM_SERVER_DEV_MODE": "1",  # /reset_prefix_cache
    "PYTHONHASHSEED": "0",
    "HF_HUB_OFFLINE": "1",
    "VLLM_CACHE_ROOT": str(W / "vllm-cache"),
    "TORCHINDUCTOR_CACHE_DIR": str(W / "vllm-cache" / "inductor"),
    "FLASHINFER_WORKSPACE_BASE": str(W / "fi_ws"),
    "VLLM_HTTP_TIMEOUT_KEEP_ALIVE": "600",
    "VLLM_ENGINE_READY_TIMEOUT_S": "3600",  # 704 GB cold load takes >600 s
    "CUDA_HOME": "/usr/local/cuda",
    "TMPDIR": str(W / "tmp"),
}


def vllm_args(mode: str, a: argparse.Namespace) -> list[str]:
    args = [
        str(MODEL),
        "--served-model-name", SERVED,
        "--port", str(VLLM_PORT),
        "--data-parallel-size", str(DP),
        "--enable-expert-parallel",
        "--kv-cache-dtype", "fp8",
        "--gpu-memory-utilization", str(a.gpu_util),
        "--max-num-batched-tokens", str(a.mbt),
        "--max-num-seqs", str(a.max_num_seqs),
        "--max-model-len", str(a.max_model_len),
        "--enable-prefix-caching",
        "--trust-remote-code",
        "--reasoning-parser", "glm45",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "glm47",
        "--enable-prompt-tokens-details",
        "--generation-config", "vllm",
    ]
    if a.mtp:
        args += ["--speculative-config", json.dumps(
            {"method": "mtp", "num_speculative_tokens": a.mtp})]
    if mode == "native":
        cfg = {
            "kv_connector": "OffloadingConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "spec_name": "CPUOffloadingSpec",
                "block_size": a.offload_block,
                "cpu_bytes_to_use": a.native_cpu_bytes,
                "eviction_policy": "lru",
            },
        }
    else:
        cfg = {
            "kv_connector": "LMCacheMPConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "lmcache.mp.host": "tcp://localhost",
                "lmcache.mp.port": LMC_PORT,
                "lmcache.mp.mq_timeout": 1500,
            },
        }
    args += ["--kv-transfer-config", json.dumps(cfg)]
    return args + a.vllm_arg


def lmcache_args(a: argparse.Namespace) -> list[str]:
    return [
        "server", "--host", "127.0.0.1", "--port", str(LMC_PORT),
        "--http-port", str(LMC_HTTP),
        "--chunk-size", str(a.chunk),
        "--separate-object-groups",
        "--l1-size-gb", str(a.l1_gb),
        "--eviction-policy", "LRU",
        "--disable-observability",
        "--eviction-trigger-watermark", "0.95",
        "--eviction-ratio", "0.05",
    ] + a.lmcache_arg


def aiperf_args(a: argparse.Namespace, out: Path, mode: str) -> list[str]:
    sm = [f"http://127.0.0.1:{VLLM_PORT}/metrics"]
    if mode == "lmcache":
        sm.append(f"http://127.0.0.1:{LMC_HTTP}/metrics")
    args = [
        "profile",
        "--url", f"http://127.0.0.1:{VLLM_PORT}",
        "--model", SERVED,
        "--endpoint-type", "chat",
        "--streaming",
        "--extra-inputs", "ignore_eos:true",
        "--use-server-token-count",
        "--random-seed", "0",
        "--export-level", "records",
        "--slice-duration", "30",
        "--no-gpu-telemetry",
        "--scenario", "inferencex-agentx-mvp",
        "--unsafe-override",
        "--tokenizer", HF_REPO,
        "--custom-dataset-type", "weka_trace",
        "--input-file", str(W / "trace" / "weka-062126-256k-dir"),
        "--max-context-length", str(a.max_context),
        "--prompt-corpus", "coding",
        "--cache-bust", "first_turn_prefix",
        "--benchmark-duration", str(a.duration),
        "--benchmark-grace-period", "60",
        "--ui", "simple",
        "--concurrency", str(a.concurrency),
        "--server-metrics", *sm,
        "--artifact-dir", str(out / "aiperf"),
    ]
    return args + a.aiperf_arg


def wait_http(url: str, timeout: float, procs: list[subprocess.Popen]) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        for p in procs:
            if p.poll() is not None:
                return False
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False


def post(url: str) -> int:
    req = urllib.request.Request(url, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.status


def free_gpus() -> list[int]:
    q = lambda arg: subprocess.run(["nvidia-smi", arg, "--format=csv,noheader"],
                                   capture_output=True, text=True, check=True).stdout.split("\n")
    busy = {u.strip() for u in q("--query-compute-apps=gpu_uuid") if u.strip()}
    out = []
    for row in q("--query-gpu=index,uuid,memory.used"):
        if not row.strip():
            continue
        i, u, mem = [x.strip() for x in row.split(",")]
        if u not in busy or int(mem.split()[0]) < 2048:
            out.append(int(i))
    return out


def start(cmd: list[str], env: dict[str, str], log: Path) -> subprocess.Popen:
    f = open(log, "w")
    return subprocess.Popen(cmd, env={**os.environ, **env}, stdout=f, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)


def stop(p: subprocess.Popen | None, grace: float = 60) -> None:
    if p is None or p.poll() is not None:
        return
    os.killpg(p.pid, signal.SIGINT)
    try:
        p.wait(grace)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()


def extra_metrics(run_dir: Path, window: dict) -> dict:
    """Throughput and interactivity over the same window as runner.score."""
    recs = []
    with open(run_dir / "aiperf" / "profile_export.jsonl") as f:
        for line in f:
            x = json.loads(line)
            me = x.get("metrics", {})
            v = lambda k: (me.get(k) or {}).get("value") if isinstance(me.get(k), dict) else None
            recs.append(v("usage_prompt_tokens") or v("input_sequence_length") or 0)
    reqs = load_requests(run_dir / "aiperf" / "profile_export.jsonl")
    for r, isl in zip(reqs, recs):
        r["isl"] = isl
    ws, we, win = window_bounds(reqs, window)
    ok = [r for r in win if not r["err"] and r["ttft"] is not None]
    span = we - ws
    isl = sum(r["isl"] for r in ok)
    osl = sum(r["osl"] for r in ok)
    inter = [r["rate"] for r in ok if r["rate"] is not None]
    return {
        "input_tok_per_s": isl / span,
        "output_tok_per_s": osl / span,
        "total_tok_per_s": (isl + osl) / span,
        "interactivity_mean": sum(inter) / len(inter) if inter else None,
        "interactivity_p50": pct(inter, 0.5),
        "ttft_mean": sum(r["ttft"] for r in ok) / len(ok) if ok else None,
        "req_per_s": len(ok) / span,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["native", "lmcache"], required=True)
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--duration", type=int, default=960)
    ap.add_argument("--max-model-len", type=int, default=262144)
    ap.add_argument("--max-context", type=int, default=262144)
    ap.add_argument("--gpu-util", type=float, default=0.96)
    ap.add_argument("--mbt", type=int, default=8192)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--mtp", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--offload-block", type=int, default=256)
    ap.add_argument("--l1-gb", type=int, default=L1_GB)
    ap.add_argument("--native-cpu-bytes", type=int, default=CPU_PER_RANK,
                    help="value passed as cpu_bytes_to_use (see meta: per-rank semantics)")
    ap.add_argument("--vllm-arg", action="append", default=[])
    ap.add_argument("--lmcache-arg", action="append", default=[])
    ap.add_argument("--aiperf-arg", action="append", default=[])
    ap.add_argument("--startup-timeout", type=int, default=3600)
    ap.add_argument("--keep-up", action="store_true", help="leave servers running (debug)")
    ap.add_argument("--reuse", action="store_true", help="use servers left up by --keep-up")
    a = ap.parse_args()

    out: Path = a.out
    out.mkdir(parents=True, exist_ok=False)
    gpus = list(range(DP)) if a.reuse else free_gpus()
    t_wait = time.time()
    while len(gpus) < DP and time.time() - t_wait < 900:  # shared host: brief foreign use
        time.sleep(15)
        gpus = free_gpus()
    if len(gpus) < DP:
        print(f"only {len(gpus)} free GPUs: {gpus}", file=sys.stderr)
        return 3
    venv = W / "venv" / "bin"
    gcc = "/opt/rh/gcc-toolset-12/root/usr/bin"  # host gcc 10 is too old for torch's C++20 JIT
    path = f"{venv}:{gcc}:/usr/local/cuda/bin:{os.environ['PATH']}"
    env = {**VLLM_ENV, "PATH": path, "CC": f"{gcc}/gcc", "CXX": f"{gcc}/g++",
           "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus[:DP]))}
    for d in ("vllm-cache/inductor", "fi_ws", "tmp"):
        (W / d).mkdir(parents=True, exist_ok=True)

    cmds = {"vllm": [str(venv / "vllm"), "serve", *vllm_args(a.mode, a)]}
    if a.mode == "lmcache":
        cmds["lmcache"] = [str(venv / "lmcache"), *lmcache_args(a)]
    ap_env = {
        "PATH": f"{W / 'venv-aiperf' / 'bin'}:{os.environ['PATH']}",
        "HF_HOME": str(W / "hf-alias"),
        "HF_HUB_OFFLINE": "1",
        "AIPERF_DATASET_CONFIGURATION_TIMEOUT": "3600",
        "AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT": "3600",
        "AIPERF_DATASET_WEKA_PARALLEL_WORKERS": "1",
        "AIPERF_DATASET_MMAP_BASE_PATH": str(out / "aiperf_mmap" / "base"),
        "AIPERF_DATASET_MMAP_CACHE_DIR": str(out / "aiperf_mmap" / "cache"),
        "PRT_LIVE_HISTORY_TOKENIZER": str(W / "hf-alias/hub/models--zai-org--GLM-5.2-FP8/snapshots/f33c6dc501ee5a2c7e35155653b1b1abbc320951"),
        "PRT_LIVE_HISTORY_LOG_DIR": str(out / "aiperf" / "prt_live_history"),
        "TMPDIR": str(W / "tmp"),
    }
    cmds["aiperf"] = [str(W / "venv-aiperf" / "bin" / "aiperf"), *aiperf_args(a, out, a.mode)]
    for sub in ("aiperf", "aiperf_mmap/base", "aiperf_mmap/cache"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    meta = {"args": vars(a) | {"out": str(out)}, "gpus": gpus[:DP], "cmds": cmds,
            "vllm_env": env, "aiperf_env": ap_env, "start": time.time()}
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))

    procs: dict[str, subprocess.Popen] = {}
    sampler = None
    rc = 1
    try:
        if a.reuse:
            # servers left up by an earlier --keep-up run with the same settings
            if not wait_http(f"http://127.0.0.1:{VLLM_PORT}/health", 30, []):
                print("--reuse: vllm not up", file=sys.stderr)
                return 4
        if a.mode == "lmcache" and not a.reuse:
            procs["lmcache"] = start(cmds["lmcache"], {**env, "LMCACHE_LOG_LEVEL": "WARNING"},
                                     out / "lmcache.log")
            if not wait_http(f"http://127.0.0.1:{LMC_HTTP}/metrics", 600, [procs["lmcache"]]):
                print("lmcache server not up", file=sys.stderr)
                return 4
        if not a.reuse:
            procs["vllm"] = start(cmds["vllm"], env, out / "vllm.log")
            t0 = time.time()
            if not wait_http(f"http://127.0.0.1:{VLLM_PORT}/health", a.startup_timeout,
                             list(procs.values())):
                print("vllm not up", file=sys.stderr)
                return 4
            meta["vllm_startup_s"] = time.time() - t0
        post(f"http://127.0.0.1:{VLLM_PORT}/reset_prefix_cache")
        status_urls = []
        if a.mode == "lmcache":
            post(f"http://127.0.0.1:{LMC_HTTP}/cache/clear")
            status_urls = [f"http://127.0.0.1:{LMC_HTTP}/status"]
        urls = [f"http://127.0.0.1:{VLLM_PORT}/metrics"]
        if a.mode == "lmcache":
            urls.append(f"http://127.0.0.1:{LMC_HTTP}/metrics")
        sampler = Sampler(out / "samples.jsonl", urls, {}, status_urls)
        sampler.start()
        p = start(cmds["aiperf"], ap_env, out / "aiperf.log")
        rc = p.wait(a.duration + 7200)
        meta["aiperf_rc"] = rc
    finally:
        if sampler:
            sampler.stop()
        if not a.keep_up:
            for k in ("vllm", "lmcache"):
                stop(procs.get(k))
        meta["end"] = time.time()
        (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))

    window = {"anchor": "first_profiling_start", "offset_s": 0,
              "length_s": a.duration, "count_by": "end"}
    res = {}
    try:
        res = score(out, window, None)
    except Exception as e:  # keep going: extra metrics may still work
        res["score_error"] = repr(e)
    try:
        res.update(extra_metrics(out, window))
    except Exception as e:
        res["extra_error"] = repr(e)
    res |= {"mode": a.mode, "concurrency": a.concurrency, "aiperf_rc": rc}
    (out / "result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0 if rc == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
