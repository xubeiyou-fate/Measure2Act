"""Advisory single-process locks for C127 orchestration entry points."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
from typing import Iterator, TextIO

if os.name == "nt":
    import msvcrt
else:
    import fcntl


@contextmanager
def exclusive_process_lock(path: Path, label: str) -> Iterator[None]:
    """Reject a second live process for the same C127 orchestration scope."""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle: TextIO = path.open("a+", encoding="utf-8")
    try:
        # msvcrt requires an existing byte, while fcntl can lock an empty file.
        if os.name == "nt":
            handle.seek(0, 2)
            if handle.tell() == 0:
                handle.write("0")
                handle.flush()
            handle.seek(0)
        try:
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"C127 {label} is already running") from error
        except OSError as error:
            if os.name == "nt":
                raise RuntimeError(f"C127 {label} is already running") from error
            raise
        yield
    finally:
        if os.name == "nt":
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
