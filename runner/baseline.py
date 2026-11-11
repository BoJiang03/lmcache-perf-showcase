# SPDX-License-Identifier: Apache-2.0
"""Cross-night regression check: has a metric shifted to a worse level?

This is the pass/fail check of PRT. There are no absolute floors and no
per-metric thresholds (Bo, 2026-10-13: "我就是不想调参"): run-to-run noise
differs by metric (TTFT p95 spread 59% across three same-config runs, decode
p10 22%), so any fixed percentage either misses real shifts on quiet metrics
or fires on noise on loud ones.

Instead each metric's nightly values form a series, and :func:`detect` asks
whether its most recent stretch ranks consistently worse than what came
before. It scans every split of the series, takes the largest standardized
Mann-Whitney statistic, and gets its p-value by permutation, so it uses only
ranks: a noisy metric is automatically harder to trigger. The one setting is
a policy, ``[regression].false_alarm_rate`` (false alarms per gated run, all
metrics together); the per-metric level is that divided by the metric count.
Everything else follows from it and the history:

- With few runs no ordering is improbable enough, so a fresh config cannot
  fail the gate until its history is long enough (no minimum-run setting).
- One bad night is never enough on its own; a shift is confirmed once
  enough nights sit on the new level.

A confirmed shift (worse or better) starts a new regime for that metric: the
next nights are tested only against runs since the shift, so a regression
fails the gate once and then becomes the baseline. Each history line stores,
per metric, the run id where its regime began (``regimes``).

History is one JSON line per valid run in ``<host.root>/prt-results/
history.jsonl``. Only runs with the same resolved run config
(:func:`config_hash`) count, so a config change is never reported as a
regression. vLLM and LMCache commits may differ (Bo, 2026-10-11,
cross-commit): a shift is labelled with which of them changed at the run
where it began, and :func:`control` checks whether the competitors, which
run the same vLLM, shifted too.

Example :func:`check` entry for one metric::

    {"value": 31.0, "median": 24.6, "change": 0.26, "n": 14,
     "p_worse": 0.002, "p_better": 1.0, "regressed": True,
     "shift": {"since": "<run id>", "before": 24.1, "after": 30.2,
               "change": 0.25, "direction": "worse", "changed": "lmcache"}}
"""

# Standard
from bisect import bisect_left
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from statistics import median
from typing import Any
import hashlib
import json
import math
import random

# First Party
from perf_regression_testing.runner.config import RunSpec

# Metrics where a larger value is better; every other compared metric is
# better when smaller.
HIGHER_IS_BETTER = frozenset({"decode_p10", "decode_p50", "mtp_mean_accept_len"})


def _ranks(xs: list[float]) -> list[float]:
    """1-based ranks of ``xs``, ties sharing their average rank."""
    order = sorted(range(len(xs)), key=xs.__getitem__)
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def _max_z(r: list[float]) -> tuple[float, int]:
    """Largest standardized Mann-Whitney U of a recent segment over the rest.

    Args:
        r: Ranks, oldest first; a high rank is a bad value.

    Returns:
        ``(z, s)``: the largest z over every split, and the index where the
        recent segment of that split starts.
    """
    n = len(r)
    best, at, tail = -math.inf, n, 0.0
    for m in range(1, n):
        tail += r[n - m]
        k = n - m
        z = (tail - m * (m + 1) / 2 - k * m / 2) / math.sqrt(k * m * (n + 1) / 12)
        if z > best:
            best, at = z, k
    return best, at


@lru_cache(maxsize=None)
def _null(n: int, perms: int) -> list[float]:
    """Sorted :func:`_max_z` of ``perms`` random orderings of ``n`` runs.

    Without ties the null depends only on ``n``, so one cache serves every
    metric and pair; the seed makes a verdict reproducible.
    """
    rng = random.Random(n)
    r = [float(i) for i in range(1, n + 1)]
    out = []
    for _ in range(perms):
        rng.shuffle(r)
        out.append(_max_z(r)[0])
    out.sort()
    return out


# RunSpec fields that change what a run measures. Timeouts, ports, validity
# patterns and the scoring of L2 reads do not, so editing them keeps the
# baseline.
_CONFIG_FIELDS = (
    "gpu_count",
    "container",
    "model_path",
    "vllm_env",
    "vllm_args",
    "services",
    "vllm_scratch",
    "files",
    "devices",
    "cap_add",
    "aiperf_env",
    "aiperf_args",
    "trace",
    "concurrency",
    "window",
)


def config_hash(spec: RunSpec) -> str:
    """Hash the resolved settings of one run that affect its results.

    Args:
        spec: Resolved run settings.

    Returns:
        12 hex chars of the SHA-256 of those fields as canonical JSON.
    """
    d = asdict(spec)
    # An optional service field left unset hashes as absent, so adding one
    # does not orphan the baselines of deployments that do not use it.
    d["services"] = [
        {k: v for k, v in svc.items() if v is not None} for svc in d["services"]
    ]
    blob = json.dumps({k: d[k] for k in _CONFIG_FIELDS}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def history_path(cfg: dict[str, Any]) -> Path:
    """Return the history file of the configured host.

    Args:
        cfg: Parsed config.

    Returns:
        ``<host.root>/prt-results/history.jsonl``.
    """
    return Path(cfg["host"]["root"]) / "prt-results" / "history.jsonl"


def load_history(path: Path) -> list[dict[str, Any]]:
    """Read every recorded run.

    Args:
        path: History file; a missing file means no runs yet.

    Returns:
        One dict per run, oldest first.
    """
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def perms_for(alpha: float) -> int:
    """Permutations for a p-value resolved well below ``alpha``.

    Args:
        alpha: Per-test false alarm level.

    Returns:
        ``ceil(100 / alpha)``: about 100 null draws beyond the critical
        value, so the Monte Carlo error is ~10% of ``alpha``.
    """
    return math.ceil(100 / alpha)


def detect(values: list[float], higher_is_better: bool, alpha: float) -> dict[str, Any]:
    """Test whether the latest values of a series shifted to another level.

    Args:
        values: One metric's values, oldest first; the last is tonight's.
        higher_is_better: True for throughput-like metrics.
        alpha: Per-test false alarm level, for each direction.

    Returns:
        ``{"n", "p_worse", "p_better", "direction", "shift_at"}``: the
        permutation p-values of the most significant split in each
        direction, ``direction`` ``"worse"``/``"better"`` when its p-value is
        at most ``alpha`` (else None), and ``shift_at`` the index where the
        shifted stretch begins (None without a shift).
    """
    n = len(values)
    out: dict[str, Any] = {
        "n": n,
        "p_worse": 1.0,
        "p_better": 1.0,
        "direction": None,
        "shift_at": None,
    }
    if n < 2:
        return out
    r = _ranks([-v if higher_is_better else v for v in values])
    null = _null(n, perms_for(alpha))
    at = {}
    for direction, ranks in (("worse", r), ("better", [n + 1 - x for x in r])):
        z, at[direction] = _max_z(ranks)
        # Ranks are exact multiples of 1/2, so 1e-9 only absorbs rounding.
        exceed = len(null) - bisect_left(null, z - 1e-9)
        out[f"p_{direction}"] = (1 + exceed) / (1 + len(null))
    best = min(("worse", "better"), key=lambda d: out[f"p_{d}"])
    if out[f"p_{best}"] <= alpha:
        out["direction"], out["shift_at"] = best, at[best]
    return out


@lru_cache(maxsize=None)
def min_runs(alpha: float) -> int:
    """Fewest runs in a regime with which :func:`detect` can report a shift.

    Args:
        alpha: Per-test false alarm level.

    Returns:
        The smallest ``n`` for which some stretch of the latest runs, each
        worse than every earlier run, reaches ``p_worse <= alpha``.
    """
    n = 2
    while not any(
        detect([*range(n - m), *range(n, n + m)], False, alpha)["direction"]
        for m in range(1, n)
    ):
        n += 1
    return n


def regime(
    history: list[dict[str, Any]],
    workload: str,
    deployment: str,
    cfg_hash: str,
    metric: str,
) -> list[dict[str, Any]]:
    """The runs of a metric's current regime, oldest first.

    Args:
        history: Output of :func:`load_history`.
        workload: Workload name.
        deployment: Deployment name.
        cfg_hash: :func:`config_hash` of the runs to use; runs recorded
            without one never match.
        metric: Summary key.

    Returns:
        The pair's runs with this config from the start of the regime that
        its latest run recorded for ``metric``; all of them when that run
        recorded none. Empty when the pair has no run with this config.
    """
    pair = [
        h
        for h in history
        if h["workload"] == workload
        and h["deployment"] == deployment
        and h.get("config_hash") == cfg_hash
    ]
    if not pair:
        return []
    start = pair[-1].get("regimes", {}).get(metric)
    ids = [h["run_id"] for h in pair]
    return pair[ids.index(start) :] if start in ids else pair


def attribute(before: dict[str, Any], after: dict[str, Any]) -> str:
    """Name the components that changed between two consecutive runs.

    Args:
        before: The last run before a shift.
        after: The first run of the shift.

    Returns:
        ``"vllm"``, ``"lmcache"``, ``"both"`` or ``"neither"``; ``"neither"``
        means a shift on identical code, i.e. host or run-to-run drift.
    """
    vllm = before["vllm_commit"] != after["vllm_commit"]
    lmcache = before["lmcache_commit"] != after["lmcache_commit"]
    if vllm and lmcache:
        return "both"
    return "vllm" if vllm else "lmcache" if lmcache else "neither"


def check(
    history: list[dict[str, Any]],
    run: dict[str, Any],
    metrics: list[str],
    alpha: float,
) -> dict[str, Any]:
    """Test one run's metrics against their regimes in the history.

    Args:
        history: Output of :func:`load_history`, without ``run``.
        run: Tonight's history record without ``regimes`` (see :func:`append`).
        metrics: Summary keys to test.
        alpha: Per-metric, per-direction false alarm level.

    Returns:
        ``{"metrics": {metric: entry}, "regimes": {metric: run id},
        "min_runs": :func:`min_runs`}``. ``regimes`` is what ``run`` must
        record. Each entry has ``value``,
        ``median`` of the regime before tonight, ``change`` (value/median - 1),
        ``n`` (regime runs including tonight), ``p_worse``, ``p_better``,
        ``regressed`` and ``shift`` (None, or ``since`` run id, the
        ``before``/``after`` medians, their ``change``, ``direction`` and
        ``changed`` per :func:`attribute`). ``value``-derived fields and
        ``regressed`` are None when tonight has no value for the metric.
    """
    out: dict[str, Any] = {"metrics": {}, "regimes": {}, "min_runs": min_runs(alpha)}
    for m in metrics:
        past = regime(
            history, run["workload"], run["deployment"], run["config_hash"], m
        )
        past = [h for h in past if h["metrics"].get(m) is not None]
        v = run["metrics"].get(m)
        if v is None:
            out["metrics"][m] = {"value": None, "regressed": None, "shift": None}
            out["regimes"][m] = past[0]["run_id"] if past else run["run_id"]
            continue
        runs = [*past, run]
        xs = [h["metrics"][m] for h in runs]
        det = detect(xs, m in HIGHER_IS_BETTER, alpha)
        med = median(xs[:-1]) if past else None
        entry = {
            "value": v,
            "median": med,
            "change": v / med - 1 if med else None,
            "n": det["n"],
            "p_worse": det["p_worse"],
            "p_better": det["p_better"],
            "regressed": det["direction"] == "worse",
            "shift": None,
        }
        start = runs[0]["run_id"]
        if det["direction"]:
            s = det["shift_at"]
            before, after = median(xs[:s]), median(xs[s:])
            entry["shift"] = {
                "since": runs[s]["run_id"],
                "before": before,
                "after": after,
                "change": after / before - 1 if before else None,
                "direction": det["direction"],
                "changed": attribute(runs[s - 1], runs[s]),
            }
            start = runs[s]["run_id"]
        out["metrics"][m] = entry
        out["regimes"][m] = start
    return out


def control(
    history: list[dict[str, Any]],
    workload: str,
    deployment: str,
    vllm_commit: str,
    metrics: list[str],
    alpha: float,
) -> dict[str, Any] | None:
    """Whether a competitor shifted on ``vllm_commit``.

    The competitors run the same vLLM image but no LMCache, so when one of
    them also shifted worse on the new vLLM, a regression points at vLLM;
    when none did, it points at LMCache. Competitors run weekly, so this
    tests the competitor's latest run on ``vllm_commit`` (not tonight's)
    against the regime it had before that run.

    Args:
        history: Output of :func:`load_history`.
        workload: Workload name.
        deployment: The competitor deployment.
        vllm_commit: vLLM commit of the run being explained.
        metrics: Summary keys to test.
        alpha: As for :func:`check`.

    Returns:
        ``{"run": run id, "check": ...}`` (see :func:`check`), or None when
        the competitor has no run on ``vllm_commit``.
    """
    pair = [
        h
        for h in history
        if h["workload"] == workload and h["deployment"] == deployment
    ]
    now = [h for h in pair if h["vllm_commit"] == vllm_commit]
    if not now:
        return None
    run = {k: v for k, v in now[-1].items() if k != "regimes"}
    earlier = history[: history.index(now[-1])]
    return {"run": run["run_id"], "check": check(earlier, run, metrics, alpha)}


def change(
    value: dict[str, Any], base: dict[str, Any], metrics: list[str]
) -> dict[str, float | None]:
    """Relative change of each metric, for side-by-side reports.

    Args:
        value: Metric -> value of the side being compared.
        base: Metric -> value of the reference side.
        metrics: Keys to compare.

    Returns:
        Metric -> ``value/base - 1``, or None when either side has no value.
    """
    return {
        m: value[m] / base[m] - 1 if value.get(m) is not None and base.get(m) else None
        for m in metrics
    }


def append(path: Path, record: dict[str, Any]) -> None:
    """Add one run to the history.

    Args:
        path: History file; created with its directory if missing.
        record: ``run_id``, ``date``, ``suite``, ``workload``, ``deployment``,
            ``vllm_commit``, ``lmcache_commit``, ``config_hash``, ``metrics``
            (summary keys) and ``regimes`` (from :func:`check`).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
