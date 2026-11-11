# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
import subprocess

# Third Party
import pytest

# First Party
from perf_regression_testing.runner import run
from perf_regression_testing.runner.config import load, resolve


def test_busy_gpus_make_the_run_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run, "pick_gpus", lambda count, wait_s: None)
    spec = resolve(load(), "agentic", "vllm_native")
    code = run.run(spec, tmp_path / "r", "img", "aiperf")
    summary = (tmp_path / "r" / "summary.json").read_text()
    assert code == run.EXIT_GPU_BUSY
    assert "busy" in summary


def _fake_smi(
    monkeypatch: pytest.MonkeyPatch, busy: list[str]
) -> list[subprocess.CompletedProcess[str]]:
    """nvidia-smi with 8 GPUs (uuid GPU-<i>) and compute processes on ``busy``."""
    calls: list[subprocess.CompletedProcess[str]] = []

    def fake(cmd: list[str], **kw: object) -> subprocess.CompletedProcess[str]:
        if "--query-compute-apps=gpu_uuid" in cmd:
            out = "".join(f"{u}\n" for u in busy)
        else:
            out = "".join(f"{i}, GPU-{i}\n" for i in range(8))
        r = subprocess.CompletedProcess(cmd, 0, out, "")
        calls.append(r)
        return r

    monkeypatch.setattr(run.subprocess, "run", fake)
    monkeypatch.setattr(run.time, "sleep", lambda s: None)
    return calls


def test_pick_gpus_skips_gpus_with_a_compute_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_smi(monkeypatch, ["GPU-0", "GPU-5", "GPU-0"])
    assert run.pick_gpus(4, wait_s=60) == [1, 2, 3, 4]


def test_pick_gpus_gives_up_when_too_few_are_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_smi(monkeypatch, [f"GPU-{i}" for i in range(5)])
    assert run.pick_gpus(4, wait_s=0) is None


def test_mooncake_clients_get_the_io_uring_seccomp_profile(tmp_path: Path) -> None:
    spec = resolve(load(), "agentic", "vllm_mooncake")
    cmds = run.build_commands(spec, [0, 1, 2, 3], "t", tmp_path, "img", "aiperf")
    for role, cmd in cmds.items():
        opts = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--security-opt"]
        if role.startswith("mooncake_client"):
            assert len(opts) == 1
            profile = Path(opts[0].removeprefix("seccomp="))
            assert "io_uring_setup" in profile.read_text()
        else:
            assert opts == []
