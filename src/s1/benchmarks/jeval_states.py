"""Real JEVal text as the `shape_sweep` state source (`shape_state_source = "jeval"`).

The synthetic source (`SyntheticStates`) builds a state from random common words. This one builds it
from the JEVal rows the evaluation runs used, so the prompts have the same content and length
structure. Rows come from `jeval.load_rows()` (pinned dataset revision); the caller injects them, so
this module never downloads anything.

How a state of N tokens is built for (seed, N, salt):
  1. Every DISTINCT state text in the pool is measured once with the run's token counter (several
     JEVal rows share one state).
  2. Candidates are ordered "closest above N first": states with at least N tokens, grouped by length
     into x1.25 bands (band 0 is N..1.25N tokens), the band order fixed, the order inside a band shuffled by
     `seed`; then the states shorter than N, shuffled. The salt picks the salt-th candidate, so
     different salts give different candidates and the text of a candidate is mostly intact (only its
     tail is cut off).
  3. A candidate with at least N tokens is cut to N tokens at a token boundary (`counter.cut` when the
     counter has one, as the file tokenizer does; otherwise at word boundaries), then re-counted and
     nudged until the count is N. A shorter candidate is first extended with other randomly chosen rows
     (never the same row twice), joined by a blank line, until it is long enough, then cut.
  4. Two candidates can cut to the same text (several JEVal sources start with the same boilerplate,
     and a 32-token cut of both is identical). Identical texts are skipped, so different salts always
     give different text. The salt wraps only after every distinct text of that length has been used.

Questions (`questions(n, salt)`): a JEVal row has ONE question and no two rows share a
`source_item_id`, so a request with Q questions gets Q real questions (choice and noul rows, never
score) drawn per request. They are real but unrelated to the (cut) state: fine for a cost sweep, not
an accuracy measurement.

The pool's own shared prefixes are kept on purpose. Real JEVal states from one source can start with the
same text, so a prefix cache can hit even though every request has a different state; `shared_prefix`
measures how much.
"""

from __future__ import annotations

import math
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .jeval import Row

if TYPE_CHECKING:
    from .shape_sweep import TokenCounter

BAND = 1.25  # states within this factor of each other count as equally close to the target
SEPARATOR = "\n\n"
MAX_FIT_ROUNDS = 8
SALT_SHIFT = 2
QUESTION_TYPES = ("choice", "noul")
_WORD = re.compile(r"\S+")


def _prefix(text: str, k: int, counter: TokenCounter) -> str:
    """The first `k` tokens of `text`: at a token boundary if the counter can cut, else at a word."""
    cut = getattr(counter, "cut", None)
    if callable(cut):
        return str(cut(text, k))
    ends = [m.end() for m in _WORD.finditer(text)]
    return text[: ends[k - 1]] if 0 < k <= len(ends) else (text if k > 0 else "")


def cut_to(text: str, tokens: int, counter: TokenCounter) -> str:
    """`text` cut to `tokens` tokens: take a prefix, re-count it and nudge until the count matches.

    A prefix cut at a token boundary can re-encode to a slightly different count (a multi-byte character
    split over tokens, a merge across the cut), hence the re-count. Keeps the closest attempt.
    """
    total = counter.count(text)
    if total <= tokens:
        return text
    best, best_err, k = text, total - tokens, tokens
    for _ in range(MAX_FIT_ROUNDS):
        cand = _prefix(text, k, counter)
        got = counter.count(cand)
        if abs(got - tokens) < best_err:
            best, best_err = cand, abs(got - tokens)
        if got == tokens:
            break
        k += tokens - got
    return best


def _lengths(counter: TokenCounter, texts: Sequence[str]) -> list[int]:
    many = getattr(counter, "count_many", None)
    return [int(n) for n in many(texts)] if callable(many) else [counter.count(t) for t in texts]


def common_prefix_chars(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def shared_prefix(texts: Sequence[str], counter: TokenCounter) -> float:
    """Mean over `texts[1:]` of the tokens each shares with the best-matching earlier text.

    This is what a prefix cache could reuse for each request after the first. 0 for unrelated texts.
    """
    if len(texts) < 2:
        return 0.0
    shared = []
    for i, t in enumerate(texts[1:], 1):
        best = max(texts[:i], key=lambda e: common_prefix_chars(e, t))
        shared.append(counter.count(t[: common_prefix_chars(best, t)]))
    return sum(shared) / len(shared)


@dataclass
class JevalStates:
    """`StateSource` and `QuestionSource` over injected JEVal rows (see the module docstring)."""

    rows: Sequence[Row]
    seed: int = 0
    _states: list[str] = field(init=False, repr=False)  # distinct state texts
    _questions: list[Mapping[str, Any]] = field(init=False, repr=False)
    _counter: TokenCounter | None = field(init=False, default=None, repr=False)
    _lens: list[int] = field(init=False, default_factory=list, repr=False)
    _order: dict[int, list[int]] = field(init=False, default_factory=dict, repr=False)
    _texts: dict[int, tuple[list[str], set[str]]] = field(init=False, default_factory=dict, repr=False)
    _tried: dict[int, int] = field(init=False, default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._states = list(dict.fromkeys(r.state for r in self.rows if r.state))
        if not self._states:
            raise ValueError("JevalStates needs at least one row with a non-empty state")
        self._questions = []
        for r in self.rows:
            q = next(iter(r.questions.values()), None)
            if r.qtype in QUESTION_TYPES and isinstance(q, dict) and len(r.questions) == 1:
                if r.qtype == "choice" and not isinstance(q.get("criteria"), dict):
                    continue
                self._questions.append(q)

    @property
    def n_states(self) -> int:
        return len(self._states)

    @property
    def n_questions(self) -> int:
        return len(self._questions)

    def lengths(self, counter: TokenCounter) -> list[int]:
        """Token count of every distinct state, measured once per counter."""
        if self._counter is not counter:
            self._counter, self._lens = counter, _lengths(counter, self._states)
            self._order.clear()
            self._texts.clear()
            self._tried.clear()
        return self._lens

    def _candidates(self, tokens: int, counter: TokenCounter) -> list[int]:
        if tokens not in self._order:
            lens, rng = self.lengths(counter), random.Random(f"{self.seed}:{tokens}:order")
            tie = {i: rng.random() for i in range(len(lens))}

            def band(i: int) -> int:
                return math.ceil(math.log(max(lens[i], tokens) / tokens, BAND) - 1e-9)

            long = sorted((i for i, n in enumerate(lens) if n >= tokens), key=lambda i: (band(i), tie[i]))
            short = sorted((i for i, n in enumerate(lens) if n < tokens), key=lambda i: tie[i])
            self._order[tokens] = long + short
        return self._order[tokens]

    def _build(self, index: int, tokens: int, counter: TokenCounter) -> str:
        """Candidate `index` as a `tokens`-token text: extended with other rows if short, then cut."""
        text = self._states[index]
        if self.lengths(counter)[index] < tokens:
            rng = random.Random(f"{self.seed}:{tokens}:{index}:extend")
            others = [i for i in range(len(self._states)) if i != index]
            rng.shuffle(others)
            for j in others:
                if counter.count(text) >= tokens:
                    break
                text += SEPARATOR + self._states[j]
        return cut_to(text, tokens, counter)

    def state(self, tokens: int, counter: TokenCounter, salt: int = 0) -> str:
        order = self._candidates(tokens, counter)
        texts, seen = self._texts.setdefault(tokens, ([], set()))
        # Candidates are consumed in order and kept only if their text is new, so the salt-th distinct
        # text never depends on the call order (`texts` and `_tried` are only a memo).
        index = salt + SALT_SHIFT  # the sweep's warm-up uses salt -2; any lower salt wraps like a high one
        tried = self._tried.get(tokens, 0)
        while len(texts) <= index and tried < len(order):
            built = self._build(order[tried], tokens, counter)
            tried += 1
            if built not in seen:
                seen.add(built)
                texts.append(built)
        self._tried[tokens] = tried
        return texts[index % len(texts)]

    def questions(self, n: int, salt: int = 0) -> dict[str, Any]:
        """`n` distinct real questions (choice / noul), drawn by (seed, n, salt)."""
        if not self._questions:
            raise ValueError("JevalStates has no choice or noul questions to draw")
        rng = random.Random(f"{self.seed}:{n}:{salt}:questions")
        if n <= len(self._questions):
            picked = rng.sample(self._questions, n)
        else:
            picked = [rng.choice(self._questions) for _ in range(n)]
        return {f"q{i}": q for i, q in enumerate(picked)}
