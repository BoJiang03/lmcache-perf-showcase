# SPDX-License-Identifier: Apache-2.0
"""Manual A/B: two images, back to back on the same host and GPUs.

For bisecting a regression the nightly baseline flagged (Bo, 2026-10-11):
the baseline spans nights and hosts, so a small delta can be drift; an A/B
run removes that by running both sides of the change in one session.

Usage (repo root, Python >= 3.11, docker on the host)::

    VLLM_COMMIT=<sha> bash perf_regression_testing/docker/build.sh serving
    bash perf_regression_testing/docker/build.sh lmcache <LMCache dir> <serving>
    python -m perf_regression_testing.ab --workload agentic \\
        --deployment vllm_lmcache --a prt-lmcache:<a> --b prt-lmcache:<b> \\
        --aiperf-image prt-aiperf:<sum> --rounds 2 --out <dir>

Any vLLM x LMCache pair works, since each side is just an image. Rounds run
A B, then B A, alternating (ABBA), so a linear drift over the session
cancels in the per-side median. Nothing is written to the nightly history.

Output in ``--out``: one run dir per run (``<i>-a`` / ``<i>-b``), plus
``ab.json`` and ``ab.md``: each side's commits and per-metric median, and
B's change against A. There is no verdict: a few runs per side are too few
for the rank test of the nightly check (runner/baseline.py), so the change
is read next to the runs' own spread.
Exit codes: 0 every run valid, 2 a run was invalid (stops the session).
"""

# Standard
from pathlib import Path
from statistics import median
from typing import Any
import argparse
import json
import sys

# First Party
from perf_regression_testing.runner import baseline, docker
from perf_regression_testing.runner.config import load, resolve
from perf_regression_testing.runner.run import EXIT_VALID, run

EXIT_AB_INVALID = 2


def order(rounds: int) -> list[str]:
    """The side of each run: ``a b``, then ``b a``, alternating per round.

    Args:
        rounds: Number of A/B pairs.

    Returns:
        ``"a"``/``"b"`` per run, in run order.
    """
    return [
        s for r in range(rounds) for s in (("a", "b") if r % 2 == 0 else ("b", "a"))
    ]


def summarize(
    summaries: dict[str, list[dict[str, Any]]], metrics: list[str]
) -> dict[str, Any]:
    """Median of each side and B's change against A.

    Args:
        summaries: ``"a"``/``"b"`` -> the ``summary.json`` of each valid run.
        metrics: ``[regression].metrics``.

    Returns:
        ``{"a": {metric: median}, "b": {...}, "compare": ...}``, where
        ``compare`` is :func:`baseline.change` of B against A.
    """
    med: dict[str, dict[str, float | None]] = {}
    for side, runs in summaries.items():
        med[side] = {}
        for m in metrics:
            xs = [s[m] for s in runs if s.get(m) is not None]
            med[side][m] = median(xs) if xs else None
    return {
        **med,
        "compare": baseline.change(med["b"], med["a"], metrics),
    }


def report_md(rep: dict[str, Any]) -> str:
    """Render ``ab.json`` as Markdown.

    Args:
        rep: The ``ab.json`` content.

    Returns:
        Markdown text.
    """
    lines = [f"# PRT A/B {rep['workload']} x {rep['deployment']}", ""]
    for side in ("a", "b"):
        im = rep["images"][side]
        lines.append(
            f"- {side.upper()}: `{im['image']}` vLLM {(im['vllm_commit'] or '?')[:9]},"
            f" LMCache {(im['lmcache_commit'] or '?')[:9]}"
        )
    lines += [
        f"- runs: {' '.join(r['run'] for r in rep['runs'])}",
        "",
        "| metric | A | B | B vs A |",
        "|---|---|---|---|",
    ]
    res = rep.get("result") or {"compare": {}}
    for m, c in res["compare"].items():
        change = "" if c is None else f"{c:+.1%}"
        lines.append(f"| {m} | {res['a'][m]} | {res['b'][m]} | {change} |")
    if rep["exit_code"] != EXIT_VALID:
        bad = rep["runs"][-1]
        lines += ["", f"STOPPED: run {bad['run']} exit {bad['exit_code']}"]
    return "\n".join(lines) + "\n"


def ab(
    workload: str,
    deployment: str,
    images: dict[str, str],
    aiperf_image: str,
    rounds: int,
    out: Path,
) -> int:
    """Run the A/B session; see the module docstring.

    Args:
        workload: Key under ``[workloads]``.
        deployment: Key under ``[deployments]``.
        images: ``"a"``/``"b"`` -> deployment image.
        aiperf_image: AIPerf image, the same for both sides.
        rounds: Number of A/B pairs.
        out: Output dir; must not exist.

    Returns:
        Exit code.
    """
    cfg = load()
    out.mkdir(parents=True)
    rep: dict[str, Any] = {
        "workload": workload,
        "deployment": deployment,
        "images": {
            side: {
                "image": img,
                "vllm_commit": docker.image_label(img, "prt.vllm.commit"),
                "lmcache_commit": docker.image_label(img, "prt.lmcache.commit"),
            }
            for side, img in images.items()
        },
        "runs": [],
    }

    def finish(code: int) -> int:
        rep["exit_code"] = code
        (out / "ab.json").write_text(json.dumps(rep, indent=1))
        (out / "ab.md").write_text(report_md(rep))
        return code

    summaries: dict[str, list[dict[str, Any]]] = {"a": [], "b": []}
    for i, side in enumerate(order(rounds)):
        run_dir = out / f"{i}-{side}"
        code = run(
            resolve(cfg, workload, deployment), run_dir, images[side], aiperf_image
        )
        rep["runs"].append({"run": run_dir.name, "side": side, "exit_code": code})
        if code != EXIT_VALID:
            return finish(EXIT_AB_INVALID)
        summaries[side].append(json.loads((run_dir / "summary.json").read_text()))
        finish(-1)
    rep["result"] = summarize(summaries, cfg["regression"]["metrics"])
    return finish(EXIT_VALID)


def main() -> None:
    """Parse arguments and run the A/B session."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workload", required=True)
    ap.add_argument("--deployment", required=True)
    ap.add_argument("--a", required=True, help="image of side A (the reference)")
    ap.add_argument("--b", required=True, help="image of side B")
    ap.add_argument("--aiperf-image", required=True)
    ap.add_argument("--rounds", type=int, default=1, help="A/B pairs, ABBA order")
    ap.add_argument("--out", required=True, type=Path, help="must not exist")
    a = ap.parse_args()
    sys.exit(
        ab(
            a.workload,
            a.deployment,
            {"a": a.a, "b": a.b},
            a.aiperf_image,
            a.rounds,
            a.out,
        )
    )


if __name__ == "__main__":
    main()
