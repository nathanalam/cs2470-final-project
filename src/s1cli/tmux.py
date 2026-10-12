"""tmux helpers: detect, list, create and attach sessions resiliently."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass

PREFIX = "s1-"


def available() -> bool:
    return shutil.which("tmux") is not None


def inside() -> bool:
    return bool(os.environ.get("TMUX"))


def _tmux(*args: str, check: bool = True) -> str:
    proc = subprocess.run(["tmux", *args], capture_output=True, text=True, check=False)
    if check and proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"tmux {' '.join(args)} failed")
    return proc.stdout


def current_session() -> str | None:
    if not inside():
        return None
    return _tmux("display-message", "-p", "#S").strip() or None


@dataclass
class Window:
    session: str
    index: str
    name: str
    attached: bool

    @property
    def target(self) -> str:
        return f"{self.session}:{self.index}"


def list_windows() -> list[Window]:
    """All windows across all sessions (empty if no tmux server is running)."""
    if not available():
        return []
    out = _tmux(
        "list-windows",
        "-a",
        "-F",
        "#{session_name}|#{window_index}|#{window_name}|#{session_attached}",
        check=False,
    )
    wins = []
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) == 4:
            wins.append(Window(parts[0], parts[1], parts[2], parts[3] not in ("0", "")))
    return wins


def s1_windows() -> list[Window]:
    """Windows created by s1 (window or session name starts with s1-)."""
    return [w for w in list_windows() if w.name.startswith(PREFIX) or w.session.startswith(PREFIX)]


def launch(name: str, shell_cmd: str) -> str:
    """Start `shell_cmd` in tmux and return the tmux target holding it.

    Outside tmux: a new detached session named `name`.
    Inside tmux:  a new window named `name` in the current session (no nesting).
    """
    if inside():
        target = _tmux(
            "new-window", "-P", "-F", "#{session_name}:#{window_index}", "-n", name, shell_cmd
        ).strip()
        return target
    _tmux("new-session", "-d", "-s", name, "-n", name, shell_cmd)
    return name


def attach(target: str) -> int:
    """Attach/switch to a tmux target. Outside tmux this blocks until detach (Ctrl-b d)."""
    if inside():
        session = target.split(":", maxsplit=1)[0]
        _tmux("switch-client", "-t", session, check=False)
        _tmux("select-window", "-t", target, check=False)
        return 0
    return subprocess.call(["tmux", "attach-session", "-t", target])


def kill(target: str) -> None:
    if ":" in target:
        _tmux("kill-window", "-t", target, check=False)
    else:
        _tmux("kill-session", "-t", target, check=False)


def exists(target: str) -> bool:
    if not available():
        return False
    if ":" in target:
        return any(w.target == target for w in list_windows())
    return (
        subprocess.run(["tmux", "has-session", "-t", target], capture_output=True, check=False).returncode
        == 0
    )
