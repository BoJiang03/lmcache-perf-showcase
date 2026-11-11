# SPDX-License-Identifier: Apache-2.0
"""Tonight's images: LMCache built FROM a vLLM image refreshed weekly.

Flow (Bo, 2026-10-06: track vLLM nightly, on failure fall back and log;
2026-10-08: refresh vLLM weekly, not nightly)::

    last-good serving younger than VLLM_REFRESH_S:
        build lmcache FROM it -> smoke -> use                      ("pinned")
    else:
        build serving@nightly -> build lmcache FROM it -> smoke -> pass: use,
        mark last-good                                             ("nightly")
           any step fails: log the error, then
        build lmcache FROM last-good serving -> smoke -> use       ("fallback")

GPUs held by someone else during a smoke say nothing about the image: the
night gets no images (``gpus_busy``), no fallback and no nightly error.

"Usable" means built AND smoke-tested (:mod:`smoke`), never built alone. The
LMCache under test is built every night either way, so LMCache nightly is
still tested when vLLM nightly is broken. If the fallback also fails, the
LMCache build itself is broken and the night has no images.

State lives in ``<host.root>/prt-images/``: ``last_good.json`` (the serving
image and its vLLM commit) and ``history.jsonl`` (one line per night, with
the error of a rejected nightly). Build logs and smoke dirs go under
``--out``.

Usage::

    python -m perf_regression_testing.runner.images \\
        --lmcache-src <clean LMCache checkout> --out <dir>

Prints and writes ``<out>/images.json``: ``serving``, ``lmcache``,
``vllm_commit``, ``vllm_source`` (``pinned``, ``nightly`` or ``fallback``),
``nightly_error``, ``gpus_busy``. Exits 0 when the night has usable images, 1 otherwise.
"""

# Standard
from pathlib import Path
from typing import Any
import argparse
import json
import subprocess
import sys
import time

# First Party
from perf_regression_testing.runner import docker
from perf_regression_testing.runner.config import PKG_DIR, load, resolve
from perf_regression_testing.runner.smoke import SMOKE_WORKLOAD, smoke

BUILD_SH = PKG_DIR / "docker" / "build.sh"
DEPLOYMENT = "vllm_lmcache"
# Rebuilt every night; anything not kept is removed by :func:`prune`.
NIGHTLY_REPOS = ("prt-serving", "prt-lmcache", "prt-mooncake")
# A new vLLM costs ~25 min of image build (mostly the fused_moe_90 JIT) plus a
# cold torch.compile and autotune, and pinning it for a week leaves LMCache as
# the only thing that changes between nights.
VLLM_REFRESH_S = 7 * 86400


class GpusBusy(Exception):
    """The smoke could not get its GPUs; nothing is known about the image."""


def build(args: list[str], log: Path) -> str:
    """Run build.sh, logging to ``log``; return the tag it printed.

    Args:
        args: build.sh arguments (target first).
        log: File that receives the build output.

    Returns:
        The image tag.

    Raises:
        RuntimeError: when the build fails; the message names the log.
    """
    with open(log, "w") as f:
        rc = subprocess.run(
            ["bash", str(BUILD_SH), *args], stdout=f, stderr=subprocess.STDOUT
        ).returncode
    lines = log.read_text(errors="replace").splitlines()
    if rc != 0:
        tail = " | ".join(lines[-5:])
        raise RuntimeError(f"build.sh {args[0]} rc={rc} ({log}): {tail}")
    # build.sh ends with the tag, then the image id.
    return lines[-2]


def _try(
    serving: str, lmcache_src: Path, out: Path, name: str, cfg: dict[str, Any]
) -> tuple[str | None, str | None]:
    """Build lmcache FROM ``serving`` and smoke-test it.

    Returns:
        (lmcache image, None) when usable, else (None, error).

    Raises:
        GpusBusy: the smoke could not get its GPUs.
    """
    try:
        lmcache = build(
            ["lmcache", str(lmcache_src), serving], out / f"build_lmcache_{name}.log"
        )
    except RuntimeError as e:
        return None, str(e)
    spec = resolve(cfg, SMOKE_WORKLOAD, DEPLOYMENT)
    r = smoke(spec, lmcache, out / f"smoke_{name}")
    if r["failures"] == ["GPUs busy"]:
        raise GpusBusy(
            f"smoke of {lmcache}: GPUs busy, fewer than {spec.gpu_count} free"
        )
    if not r["pass"]:
        return None, f"smoke of {lmcache}: {'; '.join(r['failures'])}"
    return lmcache, None


def _nightly_or_fallback(
    lmcache_src: Path,
    out: Path,
    cfg: dict[str, Any],
    last_good: dict[str, Any] | None,
    res: dict[str, Any],
) -> tuple[str | None, str | None]:
    """Try tonight's vLLM nightly, else the last-good serving image.

    Records ``vllm_source`` and the errors in ``res``, and marks a usable
    nightly as last-good.

    Returns:
        (lmcache image or None, the serving image it is built FROM).

    Raises:
        GpusBusy: a smoke could not get its GPUs.
    """
    serving = None
    try:
        serving = build(["serving"], out / "build_serving.log")
        res["nightly_serving"] = serving
        lmcache, err = _try(serving, lmcache_src, out, "nightly", cfg)
    except RuntimeError as e:
        lmcache, err = None, str(e)
    if lmcache is not None:
        res["vllm_source"] = "nightly"
        state = Path(cfg["host"]["root"]) / "prt-images" / "last_good.json"
        state.write_text(
            json.dumps(
                {
                    "serving": serving,
                    "vllm_commit": docker.image_label(serving, "prt.vllm.commit"),
                    "time": res["time"],
                },
                indent=1,
            )
        )
        return lmcache, serving
    res["nightly_error"] = err
    res["vllm_source"] = "fallback"
    if last_good is None:
        res["fallback_error"] = "no last-good serving image yet"
        return None, None
    lmcache, res["fallback_error"] = _try(
        last_good["serving"], lmcache_src, out, "fallback", cfg
    )
    return lmcache, last_good["serving"]


def tonight(lmcache_src: Path, out: Path, need_lmcache: bool = True) -> dict[str, Any]:
    """Produce tonight's images; see the module docstring.

    A suite with no LMCache pair (weekly) passes ``need_lmcache=False``: while
    vLLM is pinned it gets the pinned serving image with no LMCache build or
    smoke, so an LMCache break cannot void it. A vLLM refresh still builds and
    smokes LMCache, since that smoke is what validates the new serving image.

    Args:
        lmcache_src: Clean LMCache checkout at the commit under test.
        need_lmcache: Whether the suite runs an LMCache pair.
        out: Dir for build logs, smoke dirs and ``images.json``; created.

    Returns:
        The ``images.json`` content.
    """
    cfg = load()
    state = Path(cfg["host"]["root"]) / "prt-images"
    state.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True)
    last_good_path = state / "last_good.json"
    last_good = (
        json.loads(last_good_path.read_text()) if last_good_path.exists() else None
    )

    res: dict[str, Any] = {
        "time": time.time(),
        "nightly_error": None,
        "gpus_busy": None,
    }
    lmcache = None
    try:
        if last_good is not None and res["time"] - last_good["time"] < VLLM_REFRESH_S:
            res["vllm_source"] = "pinned"
            res["pinned_since"] = last_good["time"]
            serving = last_good["serving"]
            if not need_lmcache:
                res.update(serving=serving, vllm_commit=last_good.get("vllm_commit"))
                res["usable"] = True
                (out / "images.json").write_text(json.dumps(res, indent=1))
                return res
            lmcache, res["pinned_error"] = _try(
                serving, lmcache_src, out, "pinned", cfg
            )
        else:
            lmcache, serving = _nightly_or_fallback(
                lmcache_src, out, cfg, last_good, res
            )
    except GpusBusy as e:
        res["gpus_busy"] = str(e)
    if lmcache is not None:
        res.update(
            serving=serving,
            lmcache=lmcache,
            vllm_commit=docker.image_label(lmcache, "prt.vllm.commit"),
            lmcache_commit=docker.image_label(lmcache, "prt.lmcache.commit"),
        )
    res["usable"] = lmcache is not None
    (out / "images.json").write_text(json.dumps(res, indent=1))
    with open(state / "history.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")
    return res


def prune(keep: list[str]) -> list[str]:
    """Remove nightly images other than ``keep`` and the last-good serving.

    On h200-16 images live on the 438G root fs (containerd snapshotter), so
    old nights must not pile up. Run after the night's runs, never during.

    Args:
        keep: Image tags still needed (tonight's).

    Returns:
        The tags removed.
    """
    state = Path(load()["host"]["root"]) / "prt-images" / "last_good.json"
    keep = [*keep, json.loads(state.read_text())["serving"]] if state.exists() else keep
    tags = subprocess.run(
        ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    gone = [t for t in tags if t.split(":")[0] in NIGHTLY_REPOS and t not in keep]
    if gone:
        subprocess.run(["docker", "rmi", *gone], capture_output=True, check=True)
    return gone


def main() -> None:
    """Parse arguments and produce tonight's images."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lmcache-src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="must not exist")
    a = ap.parse_args()
    r = tonight(a.lmcache_src, a.out)
    print(json.dumps(r, indent=1))
    sys.exit(0 if r["usable"] else 1)


if __name__ == "__main__":
    main()
