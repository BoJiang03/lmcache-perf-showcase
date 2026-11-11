# SPDX-License-Identifier: Apache-2.0
"""Run a whole test suite: images, every (workload, deployment) run, report.

Usage (repo root, Python >= 3.11, docker on the host)::

    python -m perf_regression_testing.suite --suite nightly --out <dir>

Flow::

    LMCache checkout -> tip of [lmcache].branch
    -> tonight's images (runner/images.py: vLLM nightly, else last-good + error)
       (no LMCache build or smoke when no pair uses it and vLLM is pinned)
    -> mooncake image FROM tonight's serving, if a deployment needs it
    -> AIPerf image
    -> runner/run.py for each pair of [suites.<name>], one after another
    -> each valid run vs its own history (runner/baseline.py), then into history
    -> <out>/report.json and report.md

The report names the vLLM and LMCache commits actually used and whether vLLM
is pinned (the weekly vLLM), tonight's nightly or a fallback, so a
cross-night delta is not blamed on the wrong component. A run is tested
against past runs of the same config on any vLLM and LMCache commits
(:func:`baseline.check`, no per-metric thresholds); a shift is labelled with
what changed at the run where it began (vLLM, LMCache, both, or neither =
drift) and with how the competitors moved on the same vLLM
(:func:`baseline.control`). Pass or fail is a worse shift only.
Exit codes: 0 every pair valid and none regressed; 1 a gated pair regressed;
2 a pair was invalid (or no usable images); a non-gating suite returns 0
whenever every pair produced a valid result.
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
from perf_regression_testing.runner import baseline, docker, images
from perf_regression_testing.runner.config import PKG_DIR, load, resolve
from perf_regression_testing.runner.run import EXIT_VALID, run

EXIT_SUITE_FAIL, EXIT_SUITE_INVALID = 1, 2
REPORT_COLS = [
    ("decode_p10", "decode p10 tok/s"),
    ("ttft_p95", "TTFT p95 s"),
    ("preemptions", "preemptions"),
    ("external_token_share", "cache-layer token share"),
    ("l2_read_gb", "L2 read GB"),
    ("l2_disk_used_gb", "L2 disk used GB"),
]


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def update_lmcache(cfg: dict[str, Any]) -> Path:
    """Check out the tip of the LMCache branch under test in our own clone.

    Args:
        cfg: Parsed config.

    Returns:
        The checkout dir (clean, detached at the fetched tip).
    """
    lm = cfg["lmcache"]
    src = Path(cfg["host"]["root"]) / lm["checkout"]
    if not src.exists():
        _git("clone", "--no-checkout", lm["repo"], str(src))
    _git("-C", str(src), "fetch", lm["repo"], lm["branch"])
    _git("-C", str(src), "checkout", "--detach", "--force", "FETCH_HEAD")
    return src


def _build_aiperf(log: Path) -> str:
    with open(log, "w") as f:
        subprocess.run(
            ["bash", str(PKG_DIR / "docker" / "build.sh"), "aiperf"],
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return log.read_text().splitlines()[-2]


def _fmt(v: Any) -> str:
    return f"{v:.4g}" if isinstance(v, float) else "" if v is None else str(v)


def _vs_history(entry: dict[str, Any]) -> str:
    """One cell: each metric against this pair's own regime, shifts spelled out."""
    b = entry.get("vs_history")
    if b is None:
        return ""
    cell = _changes(b["metrics"])
    n = max((c.get("n") or 0 for c in b["metrics"].values()), default=0)
    if n < b["min_runs"]:
        cell += f" (n={n}, gate can fire from n={b['min_runs']})"
    else:
        cell += f" (n={n})"
    return cell + "".join(
        f"; {d} on this vLLM: " + (_changes(c["check"]["metrics"]) if c else "no run")
        for d, c in entry.get("control", {}).items()
    )


def _changes(metrics: dict[str, Any]) -> str:
    """Each metric's change vs its regime median; shifts marked and dated."""
    parts = []
    for m, c in metrics.items():
        if c.get("change") is None:
            continue
        part = f"{m} {c['change']:+.1%}"
        sh = c["shift"]
        if sh:
            mark = "!" if sh["direction"] == "worse" else ""
            part += (
                f" [{sh['direction'].upper()}{mark} {sh['change']:+.1%}"
                f" since {sh['since']}, changed: {sh['changed']}]"
            )
        parts.append(part)
    return ", ".join(parts)


def report_md(rep: dict[str, Any]) -> str:
    """Render the suite report as a Markdown table.

    Args:
        rep: The ``report.json`` content.

    Returns:
        Markdown text.
    """
    im = rep["images"]
    vllm = f"vLLM {(im.get('vllm_commit') or '?')[:9]}"
    if im.get("disk_full"):
        vllm += f" (NO IMAGES: {im['disk_full']})"
    elif im.get("gpus_busy"):
        vllm += f" (NO IMAGES: {im['gpus_busy']})"
    elif im.get("vllm_source") == "pinned":
        since = time.strftime("%Y-%m-%d", time.localtime(im["pinned_since"]))
        vllm += f" (pinned since {since})"
        if im.get("pinned_error"):
            vllm += f" (FAILED: {im['pinned_error']})"
    elif im.get("vllm_source") == "nightly":
        vllm += " (tonight's nightly, NEW vLLM)"
    elif im.get("vllm_source") == "fallback":
        vllm += f" (FALLBACK, tonight's nightly failed: {im['nightly_error']})"
    lines = [
        f"# PRT {rep['suite']} {rep['date']}",
        "",
        vllm
        + (f"; LMCache {im['lmcache_commit'][:9]}" if im.get("lmcache") else "")
        + (
            f"; Mooncake {im['mooncake_commit'][:9]}"
            f" (nightly built {im['mooncake_built']})"
            if im.get("mooncake_commit")
            else ""
        ),
        "",
        "| workload | deployment | verdict | "
        + " | ".join(c[1] for c in REPORT_COLS)
        + " | vs own history |",
        "|---|---|---|" + "---|" * len(REPORT_COLS) + "---|",
    ]
    for r in rep["runs"]:
        s = r["summary"]
        worse = [
            m
            for m, c in r.get("vs_history", {}).get("metrics", {}).items()
            if c["regressed"]
        ]
        if s["invalid"]:
            verdict = "INVALID: " + "; ".join(s["invalid"])
        elif worse:
            verdict = "REGRESSED: " + ", ".join(worse)
        else:
            verdict = "pass"
        cells = [_fmt(s.get(k)) for k, _ in REPORT_COLS] + [_vs_history(r)]
        lines.append(
            f"| {r['workload']} | {r['deployment']} | {verdict} | "
            + " | ".join(cells)
            + " |"
        )
    return "\n".join(lines) + "\n"


def suite(name: str, out: Path) -> int:
    """Run suite ``name`` into ``out``; see the module docstring.

    Args:
        name: Key under ``[suites]``.
        out: Output dir; must not exist.

    Returns:
        Exit code.
    """
    cfg = load()
    plan = cfg["suites"][name]
    out.mkdir(parents=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    rep: dict[str, Any] = {"suite": name, "date": stamp, "runs": []}

    def finish(code: int) -> int:
        rep["exit_code"] = code
        (out / "report.json").write_text(json.dumps(rep, indent=1))
        (out / "report.md").write_text(report_md(rep))
        return code

    free_gb, min_gb = docker.root_free_gb(), cfg["host"]["min_docker_free_gb"]
    if free_gb < min_gb:
        rep["images"] = {
            "usable": False,
            "disk_full": f"docker root fs {free_gb:.1f} GB free < {min_gb} GB",
        }
        return finish(EXIT_SUITE_INVALID)

    targets = {cfg["deployments"][d]["image_target"] for d in plan["deployments"]}
    src = update_lmcache(cfg)
    rep["images"] = im = images.tonight(
        src, out / "images", need_lmcache="lmcache" in targets
    )
    if not im["usable"]:
        return finish(EXIT_SUITE_INVALID)
    by_target = {"serving": im["serving"]}
    if im.get("lmcache"):
        by_target["lmcache"] = im["lmcache"]
    # A broken competitor image invalidates only its own pairs.
    target_error = {}
    if "mooncake" in targets:
        try:
            by_target["mooncake"] = mc = images.build(
                ["mooncake", im["serving"]], out / "build_mooncake.log"
            )
            for k in ("commit", "version", "built"):
                im[f"mooncake_{k}"] = docker.image_label(mc, f"prt.mooncake.{k}")
        except RuntimeError as e:
            target_error["mooncake"] = str(e)
    aiperf = _build_aiperf(out / "build_aiperf.log")
    reg = cfg["regression"]
    alpha = reg["false_alarm_rate"] / len(reg["metrics"])
    hist_path = baseline.history_path(cfg)
    history = baseline.load_history(hist_path)
    codes = []
    regressed = False
    for w in plan["workloads"]:
        for d in plan["deployments"]:
            spec = resolve(cfg, w, d)
            run_dir = out / f"{stamp}-{w}-{d}"
            if spec.image_target in target_error:
                err = target_error[spec.image_target]
                summary = {"invalid": [f"no {spec.image_target} image: {err}"]}
                rep["runs"].append({"workload": w, "deployment": d, "summary": summary})
                codes.append(EXIT_SUITE_INVALID)
                continue
            code = run(spec, run_dir, by_target[spec.image_target], aiperf)
            summary = json.loads((run_dir / "summary.json").read_text())
            entry = {"workload": w, "deployment": d, "summary": summary}
            if not summary["invalid"]:
                record = {
                    "run_id": run_dir.name,
                    "date": stamp,
                    "suite": name,
                    "workload": w,
                    "deployment": d,
                    "vllm_commit": summary["vllm_commit"],
                    "lmcache_commit": summary["lmcache_commit"],
                    "config_hash": baseline.config_hash(spec),
                    "metrics": {m: summary.get(m) for m in reg["metrics"]},
                }
                entry["vs_history"] = res = baseline.check(
                    history, record, reg["metrics"], alpha
                )
                worse = any(c["regressed"] for c in res["metrics"].values())
                regressed |= worse
                if worse:
                    entry["control"] = {
                        c: baseline.control(
                            history,
                            w,
                            c,
                            summary["vllm_commit"],
                            reg["metrics"],
                            alpha,
                        )
                        for c in cfg["suites"]["weekly"]["deployments"]
                        if c != d
                    }
                baseline.append(hist_path, {**record, "regimes": res["regimes"]})
            rep["runs"].append(entry)
            codes.append(code)
            # Partial reports survive a crash of a later pair.
            finish(-1)
    rep["pruned_images"] = images.prune(list(by_target.values()))
    if any(c != EXIT_VALID for c in codes):
        return finish(EXIT_SUITE_INVALID)
    if plan["gate"] and regressed:
        return finish(EXIT_SUITE_FAIL)
    return finish(EXIT_VALID)


def main() -> None:
    """Parse arguments and run the suite."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--suite", required=True, help="key under [suites]")
    ap.add_argument("--out", required=True, type=Path, help="must not exist")
    a = ap.parse_args()
    sys.exit(suite(a.suite, a.out))


if __name__ == "__main__":
    main()
