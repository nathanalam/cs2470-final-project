"""Small persistent state under ~/.config/s1cli: last allocation choices + tmux session registry."""

from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path
from typing import TypedDict, cast


class AllocDefaults(TypedDict, total=False):
    gpu_type: str | None
    gpus: int
    cpus: int
    mem: str
    time: str


class Defaults(TypedDict, total=False):
    partition: str
    by_partition: dict[str, AllocDefaults]


class SessionRecord(TypedDict):
    name: str
    target: str
    host: str
    srun: str
    created: int


def config_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    d = Path(base) / "s1cli"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load(name: str) -> object:
    p = config_dir() / name
    try:
        data: object = json.loads(p.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return data


def _save(name: str, data: object) -> None:
    p = config_dir() / name
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(p)


def load_defaults() -> Defaults:
    data = _load("defaults.json")
    return cast("Defaults", data) if isinstance(data, dict) else Defaults()


def save_defaults(d: Defaults) -> None:
    _save("defaults.json", d)


def hostname() -> str:
    return socket.gethostname().split(".")[0]


def load_sessions() -> list[SessionRecord]:
    data = _load("sessions.json")
    if not isinstance(data, list):
        return []
    keys = SessionRecord.__required_keys__
    return [cast("SessionRecord", s) for s in data if isinstance(s, dict) and keys <= s.keys()]


def record_session(name: str, target: str, srun: str) -> None:
    """Remember that tmux target `target` (session or session:window) holds job `name`."""
    sessions = [s for s in load_sessions() if s["name"] != name]
    sessions.append(
        SessionRecord(name=name, target=target, host=hostname(), srun=srun, created=int(time.time()))
    )
    _save("sessions.json", sessions[-50:])


def forget_session(name: str) -> None:
    _save("sessions.json", [s for s in load_sessions() if s["name"] != name])
