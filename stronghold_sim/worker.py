"""Standalone Python client and parallel pool for headless Stronghold Protocol JS workers.

Completely decoupled from PyTorch and RL training loops while preserving the exact
NDJSON + binary-replay wire protocol of the original worker.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import threading
from typing import Any, BinaryIO, Mapping, Sequence

try:
    import orjson as _orjson
except ImportError:  # pragma: no cover - optional fast JSON accelerator
    _orjson = None

import psutil

from .protocol import COOP_MODES, ROOT, SEATS_PER_COOP_GAME, WORKER_SCRIPT, expected_source
from .runtime import find_node


class WorkerError(RuntimeError):
    """Raised when a JS simulator worker fails, exits unexpectedly, or violates protocol."""


class RoomClient:
    """Synchronous client owning a single headless four-seat Stronghold Protocol JS worker."""

    def __init__(
        self,
        *,
        node_bin: str | Path | None = None,
        worker_script: str | Path | None = None,
        cwd: str | Path | None = None,
        cpu_core: int | None = None,
        log_path: str | Path | None = None,
        below_normal_priority: bool = True,
        expected_src: Mapping[str, Any] | None = None,
        verify_handshake: bool = True,
    ) -> None:
        self.node_path = find_node(node_bin)
        self.worker_script = Path(worker_script).resolve() if worker_script is not None else WORKER_SCRIPT
        self.cwd = Path(cwd).resolve() if cwd is not None else ROOT
        self._log_file: BinaryIO | None = None
        self._stderr_tail: deque[str] = deque(maxlen=64)
        self._stderr_thread: threading.Thread | None = None
        self._closed = False
        self.handshake: dict[str, Any] | None = None

        stderr_target: int | BinaryIO = subprocess.PIPE
        if log_path is not None:
            p = Path(log_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = p.open("ab")
            stderr_target = self._log_file

        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NO_WINDOW
            if below_normal_priority:
                creationflags |= subprocess.BELOW_NORMAL_PRIORITY_CLASS

        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}
        self.process = subprocess.Popen(
            [str(self.node_path), str(self.worker_script)],
            cwd=str(self.cwd),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_target,
            creationflags=creationflags,
        )

        if self._log_file is None and self.process.stderr is not None:
            self._stderr_thread = threading.Thread(
                target=self._drain_stderr,
                name=f"stronghold-stderr-{self.process.pid}",
                daemon=True,
            )
            self._stderr_thread.start()

        try:
            if cpu_core is not None:
                psutil.Process(self.process.pid).cpu_affinity([cpu_core])
            if verify_handshake:
                src = dict(expected_src) if expected_src is not None else expected_source()
                reply = self.hello()
                if (
                    reply.get("pid") != self.process.pid
                    or reply.get("source") != src
                    or reply.get("seats") != SEATS_PER_COOP_GAME
                ):
                    raise WorkerError(f"Worker handshake mismatch: {reply!r}")
                self.handshake = reply
        except BaseException:
            self.close()
            raise

    def _drain_stderr(self) -> None:
        assert self.process.stderr is not None
        try:
            for raw in self.process.stderr:
                self._stderr_tail.append(raw.decode("utf-8", errors="replace").rstrip("\r\n"))
        except OSError:
            pass

    def _stderr_summary(self) -> str:
        if not self._stderr_tail:
            return ""
        return "\n--- worker stderr ---\n" + "\n".join(self._stderr_tail)

    def _read_exact(self, size: int) -> bytes:
        assert self.process.stdout is not None
        chunks: list[bytes] = []
        remaining = size
        while remaining > 0:
            chunk = self.process.stdout.read(remaining)
            if not chunk:
                raise EOFError(
                    f"Incomplete binary payload from JS worker (missing {remaining}/{size} bytes)"
                    + self._stderr_summary()
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks) if len(chunks) > 1 else chunks[0]

    def request(self, msg: Mapping[str, Any]) -> dict[str, Any]:
        """Send one JSON request and return the parsed reply (plus binary `replay` for `examples_memory`)."""
        if self._closed:
            raise WorkerError("Worker client is closed")
        assert self.process.stdin is not None and self.process.stdout is not None
        if _orjson is not None:
            payload = _orjson.dumps(msg) + b"\n"
        else:
            payload = (json.dumps(msg, separators=(",", ":")) + "\n").encode("utf-8")
        try:
            self.process.stdin.write(payload)
            self.process.stdin.flush()
        except OSError as exc:
            raise WorkerError(
                f"Failed to write to JS worker (exit={self.process.poll()}): {exc}"
                + self._stderr_summary()
            ) from exc

        line = self.process.stdout.readline()
        if not line:
            raise WorkerError(
                f"JS worker exited unexpectedly (code={self.process.poll()})"
                + self._stderr_summary()
            )
        response = _orjson.loads(line) if _orjson is not None else json.loads(line)
        if not response.get("ok"):
            raise WorkerError(f"JS worker error: {response.get('error')}")

        if msg.get("op") == "examples_memory":
            size = response.get("replayBytes")
            limit = msg.get("memory_limit")
            if type(size) is not int or not (0 < size <= limit):
                raise ValueError(f"Invalid compressed teacher replay length: {size!r} (limit={limit!r})")
            response["replay"] = self._read_exact(size)
        return response

    def hello(self) -> dict[str, Any]:
        return self.request({"op": "hello"})

    def reset(
        self,
        seed: int,
        mode_id: str = COOP_MODES[0],
        policy_version: int | None = None,
        *,
        compact: bool = False,
        boss_id: str | None = None,
    ) -> dict[str, Any]:
        req: dict[str, Any] = {"op": "reset", "seed": seed, "modeId": mode_id}
        if policy_version is not None:
            req["policy_version"] = policy_version
        if compact:
            req["compact"] = True
        if boss_id:
            req["bossId"] = boss_id
        return self.request(req)

    def step(
        self,
        decision_id: int,
        action_key: int,
        policy_version: int | None = None,
    ) -> dict[str, Any]:
        req: dict[str, Any] = {
            "op": "step",
            "decision_id": decision_id,
            "action_key": action_key,
        }
        if policy_version is not None:
            req["policy_version"] = policy_version
        return self.request(req)

    def generate_examples(
        self,
        mode_id: str,
        seed: int,
        *,
        directory: str | Path | None = None,
        memory_limit: int = 64 * 1024 * 1024,
        turn_order: str = "serial",
    ) -> dict[str, Any]:
        if directory is None:
            return self.request(
                {
                    "op": "examples_memory",
                    "modeId": mode_id,
                    "seed": seed,
                    "memory_limit": memory_limit,
                    "turn_order": turn_order,
                }
            )
        return self.request(
            {
                "op": "examples",
                "directory": str(Path(directory).resolve()),
                "modeId": mode_id,
                "seed": seed,
                "turn_order": turn_order,
            }
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        proc = self.process
        if proc.poll() is None:
            try:
                if proc.stdin is not None:
                    proc.stdin.write(b'{"op":"close"}\n')
                    proc.stdin.flush()
                    proc.stdin.close()
            except OSError:
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        if proc.stdout is not None:
            try:
                proc.stdout.close()
            except OSError:
                pass
        if proc.stderr is not None:
            try:
                proc.stderr.close()
            except OSError:
                pass
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=1)
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def __enter__(self) -> RoomClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class RoomPool:
    """Parallel pool of independent four-seat Stronghold Protocol JS worker processes."""

    def __init__(
        self,
        num_workers: int | None = None,
        *,
        cores: Sequence[int] | None = None,
        log_dir: str | Path | None = None,
        node_bin: str | Path | None = None,
        worker_script: str | Path | None = None,
        cwd: str | Path | None = None,
        below_normal_priority: bool = True,
        expected_src: Mapping[str, Any] | None = None,
    ) -> None:
        if cores is not None:
            core_list: list[int | None] = list(cores)
            if num_workers is not None and num_workers != len(core_list):
                raise ValueError("num_workers must match len(cores) when both are given")
        else:
            count = 1 if num_workers is None else num_workers
            if count < 1:
                raise ValueError("num_workers must be >= 1")
            core_list = [None] * count

        self.clients: list[RoomClient] = []
        self.threads = ThreadPoolExecutor(max_workers=len(core_list))
        src = dict(expected_src) if expected_src is not None else expected_source()
        log_base = Path(log_dir) if log_dir is not None else None
        try:
            for slot, core in enumerate(core_list):
                lp = (log_base / f"worker_{slot}.log") if log_base is not None else None
                client = RoomClient(
                    node_bin=node_bin,
                    worker_script=worker_script,
                    cwd=cwd,
                    cpu_core=core,
                    log_path=lp,
                    below_normal_priority=below_normal_priority,
                    expected_src=src,
                    verify_handshake=False,
                )
                self.clients.append(client)
            replies = self.exchange({w: {"op": "hello"} for w in range(len(self.clients))})
            for w, row in replies.items():
                if (
                    row.get("pid") != self.clients[w].process.pid
                    or row.get("source") != src
                    or row.get("seats") != SEATS_PER_COOP_GAME
                ):
                    raise WorkerError(f"Worker {w} handshake mismatch: {row!r}")
                self.clients[w].handshake = row
        except BaseException:
            self.close()
            raise

    @property
    def processes(self) -> list[subprocess.Popen[bytes]]:
        return [c.process for c in self.clients]

    def __len__(self) -> int:
        return len(self.clients)

    def request(self, slot: int, msg: Mapping[str, Any]) -> dict[str, Any]:
        """Synchronously send a single request to worker `slot`."""
        return self.clients[slot].request(msg)

    def exchange(self, commands: Mapping[int, Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
        """Dispatch requests concurrently across the given worker slots and collect replies."""
        if not commands:
            return {}
        if len(commands) == 1:
            slot, msg = next(iter(commands.items()))
            return {slot: self.request(slot, msg)}
        futures = {slot: self.threads.submit(self.request, slot, msg) for slot, msg in commands.items()}
        return {slot: fut.result() for slot, fut in futures.items()}

    def close(self) -> None:
        try:
            for client in self.clients:
                client.close()
        finally:
            self.threads.shutdown(wait=True)

    def __enter__(self) -> RoomPool:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
