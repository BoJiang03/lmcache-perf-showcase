# SPDX-License-Identifier: Apache-2.0
"""Resolve config.toml into the concrete settings of one run.

A run is one (workload, deployment) pair. :func:`resolve` merges, in order,
the shared base (``[vllm]``, ``[aiperf]``, ``[validity]``), the deployment
and the workload, so later layers override earlier ones key by key.
"""

# Standard
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json
import tomllib

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.toml"
PKG_DIR = CONFIG_PATH.parent


def load(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Read the config file.

    Args:
        path: TOML file to read.

    Returns:
        The parsed config.
    """
    with open(path, "rb") as f:
        return tomllib.load(f)


def cli_args(args: dict[str, Any]) -> list[str]:
    """Turn an args table into command-line arguments.

    ``True`` becomes a bare flag, ``False`` is dropped, a list becomes one
    flag followed by its values, a table becomes one JSON value, anything
    else ``--key value``.

    Args:
        args: Mapping of option name (without ``--``) to value.

    Returns:
        The argument list, in the table's order.
    """
    out: list[str] = []
    for key, value in args.items():
        if value is False:
            continue
        out.append(f"--{key}")
        if value is True:
            continue
        if isinstance(value, list):
            out += [str(v) for v in value]
        elif isinstance(value, dict):
            out.append(json.dumps(value))
        else:
            out.append(str(value))
    return out


@dataclass
class Service:
    """A cache-layer container started before vLLM."""

    name: str
    command: list[str]
    args: dict[str, Any]
    env: dict[str, str]
    ready_url: str
    ready_timeout_s: float
    scratch: list[dict[str, Any]]
    # Seccomp profile path relative to the package, replacing docker's default.
    seccomp: str | None


@dataclass
class RunSpec:
    """Everything one (workload, deployment) run needs, with layers merged."""

    workload: str
    deployment: str
    image_target: str
    gpu_count: int
    gpu_busy_wait_s: float
    container: dict[str, Any]
    model_path: str
    served_name: str
    vllm_port: int
    vllm_startup_timeout_s: float
    vllm_env: dict[str, str]
    vllm_args: dict[str, Any]
    services: list[Service]
    vllm_scratch: list[dict[str, Any]]
    devices: list[str]
    cap_add: list[str]
    files: dict[str, dict[str, Any]]
    aiperf_env: dict[str, str]
    aiperf_args: dict[str, Any]
    aiperf_timeout_s: float
    trace: dict[str, Any]
    trace_dir: Path
    concurrency: int | None
    window: dict[str, Any]
    clear_endpoints: list[str]
    invalid_log_patterns: list[str]
    l2_read: dict[str, Any] | None
    raw: dict[str, Any] = field(repr=False)


def resolve(cfg: dict[str, Any], workload: str, deployment: str) -> RunSpec:
    """Merge the config layers for one (workload, deployment) pair.

    Args:
        cfg: Parsed config from :func:`load`.
        workload: Key under ``[workloads]``.
        deployment: Key under ``[deployments]``.

    Returns:
        The merged run settings.

    Raises:
        KeyError: if ``workload`` or ``deployment`` is not in the config.
    """
    w = cfg["workloads"][workload]
    d = cfg["deployments"][deployment]
    host = cfg["host"]
    services = [
        Service(
            name=s["name"],
            command=s["command"],
            args=s.get("args", {}),
            env=s.get("env", {}),
            ready_url=s["ready_url"],
            ready_timeout_s=s["ready_timeout_s"],
            scratch=s.get("scratch", []),
            seccomp=s.get("seccomp"),
        )
        for s in d.get("services", [])
    ]
    trace = w["trace"]
    if "dir" in trace:
        trace_dir = Path(host["root"]) / trace["dir"]
    else:
        trace_dir = (PKG_DIR / trace["path"]).parent
    aiperf_args = {**cfg["aiperf"]["args"], **w["aiperf"].get("args", {})}
    aiperf_args["input-file"] = str(
        trace_dir if "dir" in trace else PKG_DIR / trace["path"]
    )
    if "concurrency" in w:
        aiperf_args["concurrency"] = w["concurrency"]
    if w["aiperf"].get("server_metrics"):
        aiperf_args["server-metrics"] = [
            f"http://127.0.0.1:{cfg['vllm']['port']}/metrics",
            *d.get("server_metrics", []),
        ]
    return RunSpec(
        workload=workload,
        deployment=deployment,
        image_target=d["image_target"],
        gpu_count=host["gpu_count"],
        gpu_busy_wait_s=host["gpu_busy_wait_s"],
        container=cfg["container"],
        model_path=cfg["model"]["path"],
        served_name=cfg["model"]["served_name"],
        vllm_port=cfg["vllm"]["port"],
        vllm_startup_timeout_s=cfg["vllm"]["startup_timeout_s"],
        vllm_env={**cfg["vllm"]["env"], **d.get("vllm_env", {})},
        vllm_args={
            **cfg["vllm"]["args"],
            **d.get("vllm_args", {}),
            **w.get("vllm_args", {}),
        },
        services=services,
        vllm_scratch=d.get("vllm_scratch", []),
        devices=d.get("devices", []),
        cap_add=d.get("cap_add", []),
        files=d.get("files", {}),
        aiperf_env={**cfg["aiperf"]["env"], **w["aiperf"].get("env", {})},
        aiperf_args=aiperf_args,
        aiperf_timeout_s=cfg["aiperf"]["timeout_s"],
        trace=trace,
        trace_dir=trace_dir,
        concurrency=w.get("concurrency"),
        window=w["scoring"]["window"],
        clear_endpoints=[
            *cfg["validity"]["clear_endpoints"],
            *d.get("clear_endpoints", []),
        ],
        invalid_log_patterns=cfg["validity"]["invalid_log_patterns"],
        l2_read=d.get("l2_read"),
        raw=cfg,
    )
