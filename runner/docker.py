# SPDX-License-Identifier: Apache-2.0
"""Thin wrappers around the docker CLI for the runner's containers.

Every container gets the ``[container]`` settings of config.toml (host
network and IPC, unlimited memlock, same-path mounts), is named
``prt-<run>-<role>`` so a crashed run can be cleaned up by name, and is
labelled ``prt.run=<run>``. Containers run as the invoking host user, so
the files they write (artifacts, L2) can be wiped without root and their
processes' I/O is attributed to their own container cgroup.
"""

# Standard
from pathlib import Path
from typing import Any
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request


def _mount_args(mounts: list[dict[str, Any]]) -> list[str]:
    out = []
    for m in mounts:
        out += ["-v", f"{m['path']}:{m['path']}" + (":ro" if m.get("ro") else "")]
    return out


def run_args(
    name: str,
    run_id: str,
    image: str,
    container: dict[str, Any],
    mounts: list[dict[str, Any]],
    env: dict[str, str],
    gpus: list[int] | None,
    devices: list[str],
    cap_add: list[str],
    seccomp: Path | None = None,
) -> list[str]:
    """Build the ``docker run`` options shared by every runner container.

    Args:
        name: Container name.
        run_id: Run id, stored as label ``prt.run``.
        image: Image reference or id.
        container: The ``[container]`` table.
        mounts: Mounts in ``[container].mounts`` form, placeholders resolved.
        env: Environment variables.
        gpus: GPU indices to expose, or None for no GPU.
        devices: Host device paths to pass through (``--device``).
        cap_add: Linux capabilities to add (``--cap-add``).
        seccomp: Seccomp profile replacing docker's default, or None.

    Returns:
        ``docker run`` arguments up to and including the image.
    """
    args = [
        "docker",
        "run",
        "--name",
        name,
        "--label",
        f"prt.run={run_id}",
        f"--network={container['network']}",
        f"--ipc={container['ipc']}",
        "--ulimit",
        f"memlock={container['ulimit_memlock']}",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-e",
        "HOME=/tmp",
    ]
    if gpus is not None:
        args += ["--gpus", '"device=' + ",".join(map(str, gpus)) + '"']
    for d in devices:
        args += ["--device", d]
    for c in cap_add:
        args += ["--cap-add", c]
    if seccomp is not None:
        args += ["--security-opt", f"seccomp={seccomp}"]
    args += _mount_args(mounts)
    for k, v in env.items():
        args += ["-e", f"{k}={v}"]
    return args + [image]


def start(args: list[str], cmd: list[str], log_path: Path) -> subprocess.Popen:
    """Start a container in the foreground of a child process, logging to a file.

    Running ``docker run`` attached (not ``-d``) keeps stdout/stderr in
    ``log_path`` without a separate ``docker logs`` follower.

    Args:
        args: Output of :func:`run_args`.
        cmd: Command and arguments run inside the container.
        log_path: File that receives the container's stdout and stderr.

    Returns:
        The ``docker run`` client process.
    """
    log = open(log_path, "w")
    return subprocess.Popen(args + cmd, stdout=log, stderr=subprocess.STDOUT)


def running(name: str) -> bool:
    """Return whether container ``name`` exists and is running."""
    r = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", name],
        capture_output=True,
        text=True,
    )
    return r.returncode == 0 and r.stdout.strip() == "true"


def status(name: str) -> str | None:
    """Return container ``name``'s state (``created``, ``running``, ``exited``...).

    Args:
        name: Container name.

    Returns:
        Docker's ``State.Status``, or None when the container does not exist.
    """
    r = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Status}}", name],
        capture_output=True,
        text=True,
    )
    return r.stdout.strip() if r.returncode == 0 else None


def cgroup_dir(name: str) -> Path:
    """Return the cgroup v2 directory of running container ``name``.

    Read from the main process's ``/proc/<pid>/cgroup``, so it holds for any
    cgroup driver. Every process of the container is in it, which is what
    per-container I/O accounting (``io.stat``) needs.
    """
    out = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Pid}}", name],
        capture_output=True,
        text=True,
        check=True,
    )
    with open(f"/proc/{int(out.stdout)}/cgroup") as f:
        (line,) = (x for x in f if x.startswith("0::"))
    return Path("/sys/fs/cgroup" + line[3:].strip())


def root_free_gb() -> float:
    """Return free GB on the filesystem holding docker's data root.

    Images, build cache and container writable layers live there, not under
    host.root.
    """
    out = subprocess.run(
        ["docker", "info", "-f", "{{.DockerRootDir}}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return shutil.disk_usage(out.stdout.strip()).free / 1e9


def image_id(image: str) -> str:
    """Return the content id (``sha256:...``) of a local image."""
    out = subprocess.run(
        ["docker", "image", "inspect", "-f", "{{.Id}}", image],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def image_label(image: str, key: str) -> str | None:
    """Return label ``key`` of a local image, or None if it has none."""
    out = subprocess.run(
        ["docker", "image", "inspect", "-f", "{{json .Config.Labels}}", image],
        capture_output=True,
        text=True,
        check=True,
    )
    return (json.loads(out.stdout) or {}).get(key)


def remove(names: list[str], timeout_s: float = 600) -> list[str]:
    """Stop and remove containers, waiting up to 120 s for a clean exit.

    ``docker rm -f`` fails while a killed container is still dying (the
    LMCache server took 39 s past SIGKILL to release 800 GiB of pinned
    memory on 2026-10-11), and the leftover name then blocks the next run,
    so removal is retried until every container is gone.

    Args:
        names: Container names; ones that do not exist are skipped.
        timeout_s: Give up retrying after this many seconds.

    Returns:
        The names that still exist after ``timeout_s``.
    """
    if not names:
        return []
    subprocess.run(["docker", "stop", "-t", "120", *names], capture_output=True)
    deadline = time.time() + timeout_s
    left = names
    while True:
        subprocess.run(["docker", "rm", "-f", *left], capture_output=True)
        left = [n for n in left if status(n) is not None]
        if not left or time.time() > deadline:
            return left
        time.sleep(5)


def wait_http(url: str, timeout_s: float, name: str) -> bool:
    """Poll ``url`` until it answers 200, the container exits, or time runs out.

    Args:
        url: URL to GET.
        timeout_s: Give up after this many seconds.
        name: Container that serves ``url``; an exit ends the wait early.

    Returns:
        True when the URL answered, False otherwise.
    """
    start_t = time.time()
    deadline = start_t + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        # The container may not exist yet in the first seconds after `docker run`,
        # and stays "created" while a freshly loaded image unpacks (~100 s for
        # the 27 GB serving image on h200-16), so only an exit ends the wait.
        st = status(name)
        if st in ("exited", "dead") or (st is None and time.time() - start_t > 30):
            return False
        time.sleep(5)
    return False
