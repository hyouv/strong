"""Node.js runtime discovery and optional CPU affinity helpers."""
from __future__ import annotations

import ctypes
import os
import shutil
import struct
from pathlib import Path
from typing import Sequence

import psutil

from .protocol import ROOT


def find_node(node_bin: str | Path | None = None) -> Path:
    """Locate a Node.js executable, preferring explicit path, env var, project-local .tools, then PATH."""
    candidates: list[Path] = []
    if node_bin is not None:
        candidates.append(Path(node_bin))
    env_node = os.environ.get("STRONGHOLD_NODE")
    if env_node:
        candidates.append(Path(env_node))

    if os.name == "nt":
        candidates.append(ROOT / ".tools" / "node" / "node.exe")
    else:
        candidates.append(ROOT / ".tools" / "node" / "bin" / "node")
        candidates.append(ROOT / ".tools" / "node" / "node")

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    on_path = shutil.which("node")
    if on_path:
        return Path(on_path).resolve()

    raise FileNotFoundError(
        "Node.js executable not found. Provide node_bin, set STRONGHOLD_NODE, "
        f"install portable Node.js under {ROOT / '.tools' / 'node'}, or add node to PATH."
    )


def physical_first_cpus(available: Sequence[int] | None = None) -> list[int]:
    """Order logical CPUs so distinct physical cores are used before SMT siblings (Windows), fallback sorted."""
    if available is None:
        try:
            avail_list = sorted(psutil.Process().cpu_affinity())
        except (AttributeError, OSError):
            avail_list = list(range(os.cpu_count() or 1))
    else:
        avail_list = sorted(set(available))

    if os.name != "nt":
        return avail_list

    try:
        size = ctypes.c_ulong()
        query = ctypes.windll.kernel32.GetSystemCpuSetInformation
        query(None, 0, ctypes.byref(size), None, 0)
        if not size.value:
            return avail_list
        buf = ctypes.create_string_buffer(size.value)
        if not query(buf, size, ctypes.byref(size), None, 0):
            return avail_list
        avail_set = set(avail_list)
        first: list[int] = []
        siblings: list[int] = []
        seen: set[tuple[int, int]] = set()
        offset = 0
        while offset < size.value:
            length, record_type = struct.unpack_from("<II", buf, offset)
            if length == 0:
                break
            if record_type == 0:
                _, group, logical, core = struct.unpack_from("<IHBB", buf, offset + 8)
                cpu = 64 * group + logical
                if cpu in avail_set:
                    (siblings if (group, core) in seen else first).append(cpu)
                    seen.add((group, core))
            offset += length
        return first + siblings + sorted(avail_set - set(first + siblings))
    except Exception:
        return avail_list
