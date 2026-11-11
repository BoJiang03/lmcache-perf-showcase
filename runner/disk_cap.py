# SPDX-License-Identifier: Apache-2.0
"""Hard byte cap on a scratch dir for cache layers that have none.

vLLM's fs offload tier never evicts and has no size option (705e124c0; the
upstream cap, vllm-project/vllm#54327, is unmerged), so on 2026-10-09 it
filled the shared /raid. Containers run as the host user, so taking the write
bits off every directory stops new block files without root: the tier's store
then fails with EACCES, which it handles like ENOSPC (the block is dropped,
lookup misses, no request fails). Loads of existing files are unaffected.
"""

# Standard
from pathlib import Path
import os
import stat
import threading
import time


def allocated_bytes(path: Path) -> int:
    """Return the bytes allocated under ``path``.

    Not ``du``: it aborts (SIGABRT) when the tree changes under it, which the
    cache layer does all the time (2026-10-09). Entries that vanish during the
    walk are skipped.
    """
    total = 0
    stack = [str(path)]
    while stack:
        try:
            it = os.scandir(stack.pop())
        except FileNotFoundError:
            continue
        with it:
            for e in it:
                try:
                    if e.is_dir(follow_symlinks=False):
                        stack.append(e.path)
                    else:
                        total += e.stat(follow_symlinks=False).st_blocks * 512
                except FileNotFoundError:
                    pass
    return total


def set_dirs_writable(path: Path, writable: bool) -> None:
    """Add or remove the write bits of ``path`` and every directory under it.

    Top-down, so a directory created during the walk lands in a parent that
    is already frozen and cannot be created at all; a second freeze pass
    catches any created before its parent was reached.
    """
    for d, _, _ in os.walk(path):
        mode = stat.S_IMODE(os.stat(d).st_mode)
        os.chmod(d, mode | stat.S_IWUSR if writable else mode & ~0o222)


class DiskCap:
    """Freeze ``path`` before it holds more than ``cap_bytes``.

    A thread measures the dir every ``poll_s`` and freezes it once the next
    poll could cross the cap: the margin is twice the growth of the last
    poll interval (``du`` itself takes seconds on a full tree). It keeps
    measuring after the freeze, because stores already in flight still land.
    """

    def __init__(self, path: Path, cap_bytes: int, poll_s: float = 5.0) -> None:
        """Args:
        path: Scratch dir the cache layer writes into.
        cap_bytes: Bytes it may hold.
        poll_s: Seconds between measurements.
        """
        self.path = path
        self.cap_bytes = cap_bytes
        self.poll_s = poll_s
        self.peak_bytes = 0
        self.frozen_at: float | None = None
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        """Start measuring."""
        self._thread.start()

    def stop(self) -> dict:
        """Stop measuring and return the result.

        Returns:
            ``cap_bytes``, ``peak_bytes`` (largest measurement) and
            ``frozen_at`` (unix time, None if the cap was never approached)
            and ``error`` (why the watch died early, else None: the cap was
            then not enforced).
        """
        self._stop.set()
        self._thread.join()
        self.peak_bytes = max(self.peak_bytes, allocated_bytes(self.path))
        return {
            "cap_bytes": self.cap_bytes,
            "peak_bytes": self.peak_bytes,
            "frozen_at": self.frozen_at,
            "error": self.error,
        }

    def _loop(self) -> None:
        try:
            self._watch()
        except Exception as e:
            # Recorded, not raised: a thread's exception would only be printed.
            self.error = repr(e)

    def _watch(self) -> None:
        last = 0
        while not self._stop.wait(self.poll_s):
            used = allocated_bytes(self.path)
            self.peak_bytes = max(self.peak_bytes, used)
            margin = 2 * max(used - last, 0)
            last = used
            if self.frozen_at is None and used + margin >= self.cap_bytes:
                set_dirs_writable(self.path, False)
                set_dirs_writable(self.path, False)
                self.frozen_at = time.time()
