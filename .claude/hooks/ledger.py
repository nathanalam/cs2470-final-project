#!/usr/bin/env python3
"""The approved-GPU-spend ledger: which Modal commands may run, and how many times.

The ledger lives in the MAIN checkout (shared by every worktree) at .claude/gpu-ledger.json, gitignored.
Entries are added only by the coordinator, only for runs the user approved in chat:

    python3 .claude/hooks/ledger.py add <id> '<regex matching the command>' <uses> '<what the user approved>'
    python3 .claude/hooks/ledger.py list

guard.py calls consume(): the first entry whose regex matches the command and has uses left is decremented
and the use is logged with a timestamp.
"""

from __future__ import annotations

import fcntl
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def path() -> Path:
    here = Path(__file__).resolve().parent
    try:
        common = subprocess.run(
            ["git", "-C", str(here), "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()  # fmt: skip
        return Path(common).parent / ".claude" / "gpu-ledger.json"
    except (OSError, subprocess.CalledProcessError):
        return here.parent / "gpu-ledger.json"


def _load(p: Path) -> dict[str, list[dict[str, object]]]:
    if not p.exists():
        return {"approved": [], "log": []}
    return json.loads(p.read_text())


def _update(fn):  # noqa: ANN001, ANN202
    p = path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.seek(0)
        raw = fh.read()
        data = json.loads(raw) if raw.strip() else {"approved": [], "log": []}
        result = fn(data)
        fh.seek(0)
        fh.truncate()
        fh.write(json.dumps(data, indent=2) + "\n")
        return result


def consume(command: str) -> dict[str, object] | None:
    def take(data: dict[str, list[dict[str, object]]]) -> dict[str, object] | None:
        for entry in data["approved"]:
            if int(str(entry.get("uses", 0))) > 0 and re.search(str(entry["match"]), command):
                entry["uses"] = int(str(entry["uses"])) - 1
                data["log"].append({"id": entry["id"], "at": datetime.now().isoformat(timespec="seconds"),
                                    "command": command[:500]})  # fmt: skip
                return entry
        return None

    return _update(take)


def summary() -> str:
    data = _load(path())
    live = [f"{e['id']} ({e['uses']} left)" for e in data["approved"] if int(str(e.get("uses", 0))) > 0]
    return ", ".join(live) or "none"


def main(argv: list[str]) -> None:
    if len(argv) >= 1 and argv[0] == "list":
        print(json.dumps(_load(path()), indent=2))
        return
    if len(argv) == 5 and argv[0] == "add":
        _, entry_id, match, uses, note = argv
        re.compile(match)

        def add(data: dict[str, list[dict[str, object]]]) -> None:
            data["approved"].append({"id": entry_id, "match": match, "uses": int(uses), "note": note,
                                     "added": datetime.now().isoformat(timespec="seconds")})  # fmt: skip

        _update(add)
        print(f"added {entry_id} ({uses} uses) to {path()}")
        return
    sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
