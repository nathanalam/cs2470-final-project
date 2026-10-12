"""Typed wrappers around questionary.

questionary returns `Any`; these helpers keep the rest of the code strictly typed by
passing choice *indices* through the prompt and mapping them back to typed values.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import questionary
from questionary import Choice, Question


class Aborted(Exception):
    """The user pressed Ctrl-C / Esc in a prompt."""


def _ask(q: Question) -> object:
    ans: object = q.ask()
    if ans is None:
        raise Aborted
    return ans


def select[T](message: str, options: Sequence[tuple[str, T]], default: T | None = None) -> T:
    """Arrow-key select over (title, value) pairs; returns the chosen value."""
    if not options:
        raise ValueError("select() needs at least one option")
    choices = [Choice(title, value=i) for i, (title, _) in enumerate(options)]
    default_choice = None
    if default is not None:
        default_choice = next((c for c, (_, v) in zip(choices, options, strict=True) if v == default), None)
    idx = _ask(questionary.select(message, choices=choices, default=default_choice))
    if not isinstance(idx, int):
        raise TypeError(f"unexpected select answer {idx!r}")
    return options[idx][1]


def confirm(message: str, *, default: bool = True) -> bool:
    ans = _ask(questionary.confirm(message, default=default))
    return ans is True


def text(message: str, validate: Callable[[str], bool | str] | None = None) -> str:
    ans = _ask(questionary.text(message, validate=validate))
    if not isinstance(ans, str):
        raise TypeError(f"unexpected text answer {ans!r}")
    return ans
