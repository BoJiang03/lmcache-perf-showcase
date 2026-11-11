# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
import json

# Third Party
import pytest

# First Party
from perf_regression_testing import suite


def test_low_docker_disk_stops_the_suite_before_any_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(suite.docker, "root_free_gb", lambda: 0.2)

    def no_build(cfg: object) -> Path:
        raise AssertionError("built with a full disk")

    monkeypatch.setattr(suite, "update_lmcache", no_build)
    assert suite.suite("weekly", tmp_path / "s") == suite.EXIT_SUITE_INVALID
    rep = json.loads((tmp_path / "s" / "report.json").read_text())
    assert "0.2 GB free" in rep["images"]["disk_full"]
    assert "NO IMAGES" in (tmp_path / "s" / "report.md").read_text()
