"""Launchers for `s1 submit`: the [launch] table of a config picks Slurm or Modal (see base.py).

Workstream B adds the [launch] parser and orchestrator dispatch here; the contract lives in base.py.
"""

from .base import Launcher, LaunchTarget, Resources

__all__ = ["LaunchTarget", "Launcher", "Resources"]
