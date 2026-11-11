# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
from typing import Any
import json
import time

# Third Party
import pytest

# First Party
from perf_regression_testing.runner import images
from perf_regression_testing.runner.config import load


@pytest.fixture
def host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Real config with host.root in tmp; docker builds and smokes faked."""
    cfg = load()
    cfg["host"]["root"] = str(tmp_path)
    calls: dict[str, Any] = {"builds": [], "smoke_failures": []}

    def build(args: list[str], log: Path) -> str:
        calls["builds"].append(args[0])
        return "prt-serving:new" if args[0] == "serving" else f"prt-lmcache:{args[2]}"

    def smoke(spec: Any, image: str, out: Path) -> dict[str, Any]:
        f = calls["smoke_failures"]
        return {"pass": not f, "failures": f}

    monkeypatch.setattr(images, "load", lambda: cfg)
    monkeypatch.setattr(images, "build", build)
    monkeypatch.setattr(images, "smoke", smoke)
    monkeypatch.setattr(images.docker, "image_label", lambda image, key: image)
    calls["state"] = tmp_path / "prt-images"
    calls["state"].mkdir()
    return calls


def _last_good(host: dict[str, Any], age_s: float) -> None:
    (host["state"] / "last_good.json").write_text(
        json.dumps({"serving": "prt-serving:old", "time": time.time() - age_s})
    )


def test_young_last_good_is_pinned_without_a_serving_build(
    host: dict[str, Any], tmp_path: Path
) -> None:
    _last_good(host, 3 * 86400)
    res = images.tonight(tmp_path, tmp_path / "out")
    assert res["vllm_source"] == "pinned"
    assert res["usable"] and res["serving"] == "prt-serving:old"
    assert host["builds"] == ["lmcache"]


def test_week_old_last_good_rebuilds_the_nightly(
    host: dict[str, Any], tmp_path: Path
) -> None:
    _last_good(host, 8 * 86400)
    res = images.tonight(tmp_path, tmp_path / "out")
    assert res["vllm_source"] == "nightly"
    assert res["serving"] == "prt-serving:new"
    lg = json.loads((host["state"] / "last_good.json").read_text())
    assert lg["serving"] == "prt-serving:new"


def test_busy_gpus_give_no_images_and_blame_no_nightly(
    host: dict[str, Any], tmp_path: Path
) -> None:
    _last_good(host, 8 * 86400)
    host["smoke_failures"] = ["GPUs busy"]
    res = images.tonight(tmp_path, tmp_path / "out")
    assert not res["usable"]
    assert res["gpus_busy"] and res["nightly_error"] is None
    assert "fallback_error" not in res
    lg = json.loads((host["state"] / "last_good.json").read_text())
    assert lg["serving"] == "prt-serving:old"


def test_suite_without_lmcache_reuses_pinned_serving_unbuilt(
    host: dict[str, Any], tmp_path: Path
) -> None:
    _last_good(host, 3 * 86400)
    host["smoke_failures"] = ["lmcache not up"]
    res = images.tonight(tmp_path, tmp_path / "out", need_lmcache=False)
    assert res["usable"] and res["serving"] == "prt-serving:old"
    assert res["vllm_source"] == "pinned"
    assert host["builds"] == []


def test_suite_without_lmcache_still_smokes_a_vllm_refresh(
    host: dict[str, Any], tmp_path: Path
) -> None:
    _last_good(host, 8 * 86400)
    res = images.tonight(tmp_path, tmp_path / "out", need_lmcache=False)
    assert res["vllm_source"] == "nightly"
    assert host["builds"] == ["serving", "lmcache"]
