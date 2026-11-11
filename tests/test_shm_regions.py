# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path

# Third Party
import pytest

# First Party
from perf_regression_testing.runner import run
from perf_regression_testing.runner.run import remove_shm_regions


def test_only_regions_this_run_created_are_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shm = tmp_path / "shm"
    shm.mkdir()
    ours, other = shm / "vllm_offload_ab-12.mmap", shm / "vllm_offload_cd-34.mmap"
    ours.touch()
    other.touch()
    log = (
        f"[shared_offload_region.py:142] Created mmap file /dev/shm/{ours.name}\n"
        "[shared_offload_region.py:128] Opened existing mmap file "
        f"/dev/shm/{other.name}\n"
    )
    (tmp_path / "vllm.log").write_text(log)
    monkeypatch.setattr(run, "SHM_DIR", shm)
    assert remove_shm_regions(tmp_path) == [ours.name]
    assert not ours.exists() and other.exists()


def test_no_log_removes_nothing(tmp_path: Path) -> None:
    assert remove_shm_regions(tmp_path) == []
