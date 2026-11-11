# SPDX-License-Identifier: Apache-2.0
# Standard
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import json
import threading
import time

# First Party
from perf_regression_testing.runner.sampler import Sampler

CLIENT = """# HELP mooncake_ssd_write_bytes_total x
mooncake_ssd_write_bytes_total{tier="ssd"} 100
mooncake_process_rss_bytes 7
"""
STATUS = json.dumps(
    {
        "storage_manager": {
            "l1_manager": {"memory_used_bytes": 800, "total_object_count": 4},
            "l2_eviction_controller": {"adapters": [{"total_bytes_used": 1500}]},
        }
    }
)
MASTER = """master_offload_failed_total 3
master_evicted_key_count 5
"""


def _serve(text: str) -> tuple[HTTPServer, str]:
    class H(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            body = text.encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}/metrics"


def test_mooncake_counters_are_recorded_and_summed_over_clients(
    tmp_path: Path,
) -> None:
    servers = [_serve(CLIENT), _serve(CLIENT), _serve(MASTER)]
    out = tmp_path / "samples.jsonl"
    s = Sampler(out, [u for _, u in servers], {}, [], interval_s=0.05)
    s.start()
    time.sleep(0.1)
    s.stop()
    for srv, _ in servers:
        srv.shutdown()
    rec = json.loads(out.read_text().splitlines()[0])
    assert rec["mooncake_ssd_write_bytes_total"] == 200.0
    assert rec["master_offload_failed_total"] == 3.0
    assert rec["master_evicted_key_count"] == 5.0
    assert "mooncake_process_rss_bytes" not in rec


def test_lmcache_status_is_sampled_every_nth_tick(tmp_path: Path) -> None:
    srv, url = _serve(STATUS)
    out = tmp_path / "samples.jsonl"
    s = Sampler(out, [], {}, [url], interval_s=0.02, status_every=3)
    s.start()
    time.sleep(0.15)
    s.stop()
    srv.shutdown()
    recs = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(recs) >= 4
    assert recs[0]["lmcache_status/l1_used_bytes"] == 800
    assert recs[0]["lmcache_status/l1_objects"] == 4
    assert recs[0]["lmcache_status/l2_used_bytes"] == 1500
    assert [("lmcache_status/l1_used_bytes" in r) for r in recs[:4]] == [
        True,
        False,
        False,
        True,
    ]
