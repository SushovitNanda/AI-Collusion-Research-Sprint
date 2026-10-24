"""Random per-run relabelling of firm letters for everything a monitor sees.

The ring is always A-B in the logs. Without relabelling a judge could guess
the first alphabetical pair and look good at detection and pair
identification. perm_for gives a permutation of the run's firms seeded by the
run id; relabel_text applies it to free text; unrelabel_pair maps a pair name
back to true letters.
"""
from __future__ import annotations

import hashlib
import random
import re

# A standalone capital letter used as a firm name: not part of a word ("Bid",
# "AND"), not part of an abbreviation ("A.I.").
_LETTER = re.compile(r"(?<![A-Za-z])(?<![A-Za-z]\.)([A-D])(?![A-Za-z])(?!\.[A-Za-z])")


def perm_for(run_id: str, firms) -> dict[str, str]:
    """true letter -> display letter, a permutation of `firms` seeded by sha256(run_id)."""
    firms = list(firms)
    seed = int(hashlib.sha256(run_id.encode()).hexdigest(), 16)
    display = list(firms)
    random.Random(seed).shuffle(display)
    return dict(zip(firms, display))


def inverse(perm: dict[str, str]) -> dict[str, str]:
    return {v: k for k, v in perm.items()}


def relabel_text(text: str, perm: dict[str, str]) -> str:
    """Replace firm letters in free text. A single callback substitution means
    A->B and B->A never collide (equivalent to a two-pass placeholder swap)."""
    if not text:
        return text
    return _LETTER.sub(lambda m: perm.get(m.group(1), m.group(1)), text)


def relabel_pair(pair: str, perm: dict[str, str]) -> str:
    i, j = pair.split("-")
    return "-".join(sorted((perm[i], perm[j])))


def unrelabel_pair(pair: str, perm: dict[str, str]) -> str:
    inv = inverse(perm)
    i, j = pair.split("-")
    return "-".join(sorted((inv[i], inv[j])))


def display_pairs(firms, perm) -> list[str]:
    """All pairs of the run in display letters, sorted."""
    disp = sorted(perm[f] for f in firms)
    return [f"{a}-{b}" for k, a in enumerate(disp) for b in disp[k + 1:]]
