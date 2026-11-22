"""Stage model code on the Modal `s1-models` volume: the Modal twin of scripts/fetch_code.sh.

    uv run --group modal python -m s1.launch.modal_stage fetch-code jaredpalmer/kev \\
        2074cc44256200bb37540103fa4572901faa10f3 kev-code     # -> s1-models:/kev-code = $MODELS_DIR/kev-code

`fetch_code(repo, revision, dirname)` runs on a CPU container (the default runtime's CPU image, which
has git): clone https://github.com/<repo>.git into /models/<dirname> unless it is already a checkout,
fetch the commit, check it out detached, and refuse the result unless HEAD is that commit and the tree
is clean (the check backends/gpu.py's check_git_commit makes before importing it). Then the volume is
committed. Weights are `s1 download` (modal_app.download). Read the code before running anything from it.

`modal` is imported only inside the functions that talk to Modal, as in modal_app.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TypedDict

from .modal_images import MODELS_MOUNT

APP_NAME = "s1-stage"
VOLUME = "s1-models"  # modal_app.VOLUMES[MODELS_MOUNT]
CPU, MEMORY_MIB, TIMEOUT_S = 2.0, 4096, 1800
_REPO = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
_SHA = re.compile(r"[0-9a-f]{40}")
_DIRNAME = re.compile(r"[A-Za-z0-9_.][A-Za-z0-9_.-]*")


class FetchReport(TypedDict):
    repo: str
    revision: str
    head: str
    dest: str
    dirty: bool


def check_args(repo: str, revision: str, dirname: str | None = None) -> str:
    """Validate the arguments locally (before any container starts); returns the directory name."""
    if not _REPO.fullmatch(repo) or ".." in repo:
        raise ValueError(f"repo must be a GitHub <owner>/<name>, got {repo!r}")
    if not _SHA.fullmatch(revision):
        raise ValueError(f"revision must be a full 40-character commit SHA (lowercase hex), got {revision!r}")
    name = dirname or repo.split("/", 1)[1]
    if not _DIRNAME.fullmatch(name) or name in (".", ".."):
        raise ValueError(f"dirname must be one plain directory name under {MODELS_MOUNT}, got {name!r}")
    return name


def fetch_report(text: str) -> FetchReport:
    """Parse and check what `_remote_fetch_code` returned."""
    raw = json.loads(text)
    if not isinstance(raw, dict):
        raise TypeError("remote fetch report must be an object")
    return {
        "repo": str(raw["repo"]),
        "revision": str(raw["revision"]),
        "head": str(raw["head"]),
        "dest": str(raw["dest"]),
        "dirty": bool(raw["dirty"]),
    }


def verify(report: FetchReport) -> FetchReport:
    """Refuse a checkout that is not at the pinned commit or has local changes."""
    if report["head"] != report["revision"]:
        raise RuntimeError(f"{report['dest']} is at {report['head']}, not the pinned {report['revision']}")
    if report["dirty"]:
        raise RuntimeError(f"{report['dest']} has local changes; the pinned code must be unmodified")
    return report


# ---- talking to Modal (imports modal lazily) ----


def fetch_code(repo: str, revision: str, dirname: str | None = None) -> FetchReport:
    """Check out <repo>@<revision> into s1-models:/<dirname> on a CPU container; raises unless verified."""
    name = check_args(repo, revision, dirname)
    from . import modal_app, modal_images  # noqa: PLC0415  (modal_app is the launcher; keep imports light)

    modal_app._check_python()
    modal = modal_app._modal()
    app = modal.App(APP_NAME)
    fn = app.function(
        image=modal_images.image(modal, openjev=False),
        cpu=CPU,
        memory=MEMORY_MIB,
        timeout=TIMEOUT_S,
        volumes={MODELS_MOUNT: modal.Volume.from_name(VOLUME, create_if_missing=True)},
        serialized=True,
        include_source=False,
    )(_remote_fetch_code)
    with modal.enable_output(), app.run():
        report = verify(fetch_report(fn.remote(repo, revision, name)))
    print(f"{repo}@{report['head']} -> s1-models:{report['dest']} ($MODELS_DIR/{name})")
    return report


# ---- what runs in the container ----

Git = Callable[..., str]


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def checkout(repo: str, revision: str, dest: Path, git: Git = _git) -> FetchReport:
    """fetch_code.sh's steps at `dest` (clone unless it is a checkout; fetch; detached checkout)."""
    if not (dest / ".git").is_dir():
        if dest.exists() and any(dest.iterdir()):
            raise FileExistsError(f"{dest} exists and is not a git checkout; refusing to overwrite it")
        git("clone", "--quiet", f"https://github.com/{repo}.git", str(dest))
    git("-C", str(dest), "fetch", "--quiet", "origin", revision)
    git("-C", str(dest), "-c", "advice.detachedHead=false", "checkout", "--quiet", revision)
    head = git("-C", str(dest), "rev-parse", "HEAD")
    dirty = bool(git("-C", str(dest), "status", "--porcelain"))
    return {"repo": repo, "revision": revision, "head": head, "dest": str(dest), "dirty": dirty}


def _remote_fetch_code(repo: str, revision: str, dirname: str) -> str:
    from . import modal_app  # noqa: PLC0415

    modal = modal_app._modal()
    report = checkout(repo, revision, Path(MODELS_MOUNT) / check_args(repo, revision, dirname))
    modal.Volume.from_name(VOLUME).commit()
    return json.dumps(report)


# ---- `python -m s1.launch.modal_stage fetch-code <repo> <revision> [dirname]` ----


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m s1.launch.modal_stage", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch-code", help="check out a GitHub repo at a commit into the s1-models volume")
    f.add_argument("repo", help="GitHub <owner>/<name>")
    f.add_argument("revision", help="full 40-character commit SHA")
    f.add_argument(
        "dirname", nargs="?", default=None, help="directory under $MODELS_DIR (default: repo name)"
    )
    return p


def main(argv: Sequence[str] | None = None, *, run: Callable[..., object] = fetch_code) -> None:
    args = parser().parse_args(argv)
    try:
        check_args(args.repo, args.revision, args.dirname)
    except ValueError as e:
        parser().error(str(e))
    run(args.repo, args.revision, args.dirname)


if __name__ == "__main__":
    main(sys.argv[1:])
