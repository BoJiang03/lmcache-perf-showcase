# SPDX-License-Identifier: Apache-2.0
# First Party
from perf_regression_testing import ab

METRICS = ["decode_p10", "ttft_p95"]


def test_order_alternates_abba() -> None:
    assert ab.order(1) == ["a", "b"]
    assert ab.order(2) == ["a", "b", "b", "a"]


def test_summarize_reports_b_against_a() -> None:
    res = ab.summarize(
        {
            "a": [
                {"decode_p10": 40.0, "ttft_p95": 3.0},
                {"decode_p10": 42.0, "ttft_p95": 3.2},
            ],
            "b": [
                {"decode_p10": 41.0, "ttft_p95": 4.0},
                {"decode_p10": 41.0, "ttft_p95": 4.2},
            ],
        },
        METRICS,
    )
    assert res["a"] == {"decode_p10": 41.0, "ttft_p95": 3.1}
    assert res["compare"]["decode_p10"] == 0.0
    assert round(res["compare"]["ttft_p95"], 3) == round(4.1 / 3.1 - 1, 3)
