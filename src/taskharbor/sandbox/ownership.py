"""Exclusive OS ownership for a single manager's persistent storage.

The lock file is never removed: replacing its inode would let separate processes
lock different files with the same name and both believe they own the directory.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import BinaryIO

from taskharbor.sandbox.storage import checked_path


class DeploymentLock:
    """A nonblocking cross-process lock, released automatically on process exit."""

    def __init__(self, root: str | Path):
        self.root = Path(os.path.abspath(root))
        self._file: BinaryIO | None = None
        self._guard = threading.Lock()

    def acquire(self) -> DeploymentLock:
        with self._guard:
            if self._file is not None:
                return self
            control = checked_path(self.root, "control")
            control.mkdir(parents=True, exist_ok=True)
            path = checked_path(control, "manager.lock")
            handle = path.open("a+b")
            try:
                # msvcrt locks a byte range; create that byte before locking it.
                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                handle.close()
                raise RuntimeError(
                    "Cannot acquire exclusive sandbox ownership: another manager may be "
                    "using this STORAGE_DIR. Reuse get_manager() or configure an independent "
                    "STORAGE_DIR for each deployment."
                ) from exc
            except BaseException:
                handle.close()
                raise
            self._file = handle
            return self

    def release(self) -> None:
        with self._guard:
            handle = self._file
            if handle is None:
                return
            self._file = None
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()

    def __enter__(self) -> DeploymentLock:
        return self.acquire()

    def __exit__(self, *args) -> None:
        self.release()
