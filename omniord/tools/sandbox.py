"""Isolated subprocess sandbox for running generated code with a hard timeout.

Code runs in a fresh ``python -I`` interpreter (isolated mode: no user site,
environment ignored) in a throwaway working directory, with a minimal
environment and a wall-clock timeout enforced by ``asyncio``. A run that
exceeds the timeout is killed and reported as ``timed_out``.

The sandbox is defense-in-depth alongside the static AST check — it is not a
full security boundary (no OS-level isolation), so untrusted code should always
pass :mod:`omniord.tools.ast_checker` first.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
import time

from pydantic import BaseModel

_MAX_OUTPUT = 100_000


class SandboxResult(BaseModel):
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool
    duration: float

    @property
    def ok(self) -> bool:
        return not self.timed_out and self.returncode == 0


def _cap(raw: bytes) -> str:
    text = raw.decode("utf-8", errors="replace")
    if len(text) > _MAX_OUTPUT:
        return text[:_MAX_OUTPUT] + "\n... (output truncated)"
    return text


class Sandbox:
    """Runs Python source in an isolated subprocess."""

    def __init__(self, timeout: float = 30.0) -> None:
        self.timeout = timeout

    async def run(self, source: str, *, timeout: float | None = None) -> SandboxResult:
        limit = timeout if timeout is not None else self.timeout
        workdir = tempfile.mkdtemp(prefix="omniord-sbx-")
        env = {"PATH": os.environ.get("PATH", ""), "HOME": workdir}
        start = time.perf_counter()
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            "-c",
            source,
            cwd=workdir,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,
        )
        timed_out = False
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=limit)
            returncode = proc.returncode
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            stdout, stderr, returncode, timed_out = b"", b"", None, True
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        return SandboxResult(
            returncode=returncode,
            stdout=_cap(stdout),
            stderr=_cap(stderr),
            timed_out=timed_out,
            duration=time.perf_counter() - start,
        )
