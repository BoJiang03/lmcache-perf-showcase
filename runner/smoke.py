# SPDX-License-Identifier: Apache-2.0
"""Smoke test of a deployment image: does it serve, and does the cache layer hit?

An image is "usable" (eligible as last-good, see :mod:`images`) only after
this passes. Steps, about the cost of one vLLM start:

1. Start the deployment's services and vLLM from the image, exactly as a run
   does (:func:`run.start_servers`).
2. Send a few short chat requests; each must return HTTP 200 with text.
3. If the deployment has a cache layer (``kv-transfer-config``): send one long
   prompt, reset vLLM's own prefix cache, send it again, and require that
   ``vllm:prompt_tokens_by_source{source="external_kv_transfer"}`` grew, i.e.
   the second prefill was served by the cache layer.
4. vllm.log, up to teardown, must not contain the invalid-run patterns or a
   Traceback.

Usage::

    python -m perf_regression_testing.runner.smoke \\
        --deployment vllm_lmcache --image prt-lmcache:<tag> --out <dir>

Writes ``<out>/smoke.json``; exits 0 on pass, 1 otherwise.
"""

# Standard
from pathlib import Path
from typing import Any
import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request

# First Party
from perf_regression_testing.runner import docker
from perf_regression_testing.runner.config import RunSpec, load, resolve
from perf_regression_testing.runner.run import (
    build_commands,
    pick_gpus,
    post,
    prepare_scratch,
    remove_shm_regions,
    start_servers,
    wipe_scratch,
)

# Any workload works: only the deployment's servers are started. agentic adds
# no vLLM args of its own, so the servers match the deployment's base.
SMOKE_WORKLOAD = "agentic"
CHAT_REQUESTS = 3
# Several LMCache chunks (1024 tokens) long, so a hit is unambiguous.
LONG_PROMPT = " ".join(
    f"Line {i}: the quick brown fox jumps over the lazy dog." for i in range(600)
)
EXTERNAL = re.compile(
    r'^vllm:prompt_tokens_by_source(?:_total)?\{[^}]*source="external_kv_transfer"[^}]*\}\s+(\S+)',
    re.M,
)
LOG_PATTERNS_EXTRA = ["Traceback"]


def _request(url: str, body: dict[str, Any]) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def _metrics(base: str) -> str:
    with urllib.request.urlopen(f"{base}/metrics", timeout=10) as r:
        return r.read().decode()


def _external_tokens(text: str) -> float:
    return sum(float(v) for v in EXTERNAL.findall(text))


def check_serving(spec: RunSpec, out: Path) -> list[str]:
    """Run steps 2 and 3 against servers that are up.

    Args:
        spec: Resolved run settings of the deployment under test.
        out: Smoke dir; a failed cache-hit check saves vLLM's ``/metrics``
            there as ``vllm_metrics.txt``.

    Returns:
        Failure reasons; empty when both steps pass.
    """
    base = f"http://127.0.0.1:{spec.vllm_port}"
    failures = []
    for i in range(CHAT_REQUESTS):
        try:
            r = _request(
                f"{base}/v1/chat/completions",
                {
                    "model": spec.served_name,
                    "messages": [
                        {"role": "user", "content": f"Count from 1 to {i + 3}."}
                    ],
                    "max_tokens": 32,
                },
            )
            msg = r["choices"][0]["message"]
            if not (msg.get("content") or msg.get("reasoning")):
                failures.append(f"chat {i}: empty response")
        except (urllib.error.URLError, TimeoutError, KeyError) as e:
            failures.append(f"chat {i}: {e}")
    if "kv-transfer-config" not in spec.vllm_args:
        return failures
    body = {"model": spec.served_name, "prompt": LONG_PROMPT, "max_tokens": 1}
    try:
        _request(f"{base}/v1/completions", body)
        # Stores are asynchronous; give the cache layer time to finish.
        time.sleep(10)
        post(f"{base}/reset_prefix_cache")
        before = _external_tokens(_metrics(base))
        _request(f"{base}/v1/completions", body)
        after = _metrics(base)
        gained = _external_tokens(after) - before
    except (urllib.error.URLError, TimeoutError) as e:
        return failures + [f"cache hit check: {e}"]
    if gained <= 0:
        (out / "vllm_metrics.txt").write_text(after)
        failures.append(
            "cache hit check: no external_kv_transfer tokens on a repeated prompt"
        )
    return failures


def smoke(spec: RunSpec, image: str, out: Path) -> dict[str, Any]:
    """Run the smoke test of ``image``; see the module docstring.

    Args:
        spec: Resolved run settings of the deployment under test.
        image: Deployment image.
        out: Output dir; must not exist.

    Returns:
        ``{"pass": bool, "failures": [...], "image": ..., "vllm_commit": ...}``,
        also written to ``out/smoke.json``.
    """
    run_id = out.name
    out.mkdir(parents=True)
    result: dict[str, Any] = {
        "image": image,
        "image_id": docker.image_id(image),
        "vllm_commit": docker.image_label(image, "prt.vllm.commit"),
        "lmcache_commit": docker.image_label(image, "prt.lmcache.commit"),
        "deployment": spec.deployment,
        "failures": [],
    }
    failures = result["failures"]
    gpus = pick_gpus(spec.gpu_count, spec.gpu_busy_wait_s)
    if gpus is None:
        failures.append("GPUs busy")
    elif reason := prepare_scratch(spec):
        failures.append(reason)
    else:
        cmds = build_commands(spec, gpus, run_id, out, image, image)
        del cmds["aiperf"]
        try:
            reason = start_servers(spec, cmds, out, run_id)
            failures += [reason] if reason else check_serving(spec, out)
            # Read before teardown: vLLM nightly 242e4213f logs a Traceback
            # (psutil.ZombieProcess) on SIGTERM, which is not a serving failure.
            vllm_log = out / "vllm.log"
            log = vllm_log.read_text(errors="replace") if vllm_log.exists() else ""
        finally:
            docker.remove([f"prt-{run_id}-{r}" for r in cmds])
            remove_shm_regions(out)
            wipe_scratch(spec)
        for pat in [*spec.invalid_log_patterns, *LOG_PATTERNS_EXTRA]:
            if pat in log:
                failures.append(f"vllm.log contains {pat!r}")
    result["pass"] = not failures
    (out / "smoke.json").write_text(json.dumps(result, indent=1))
    return result


def main() -> None:
    """Parse arguments and run the smoke test."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--deployment", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True, type=Path, help="must not exist")
    a = ap.parse_args()
    r = smoke(resolve(load(), SMOKE_WORKLOAD, a.deployment), a.image, a.out)
    print(json.dumps(r, indent=1))
    sys.exit(0 if r["pass"] else 1)


if __name__ == "__main__":
    main()
