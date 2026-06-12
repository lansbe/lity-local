from __future__ import annotations

import difflib
from dataclasses import dataclass

# Small local models routinely miss the EXACT text of a SEARCH block — a
# trailing space, a tab-vs-spaces indent, one recopied word. Each miss used to
# burn an agent step (and a slice of the failure budget) on a dead-end "not
# found". These matchers recover the obviously-intended block, in order of
# decreasing strictness, and never guess when the result would be ambiguous.

# Beyond this similarity, a unique difflib window is accepted as the intended
# block; below it we only USE the window to tell the model what to copy.
_FUZZY_ACCEPT = 0.95
# Gap required between the best and second-best window — equal scores mean the
# model's block matches two places, which is exactly when we must NOT guess.
_FUZZY_GAP = 0.01
# Whole-file difflib scans get slow on very large files; the cheap line-based
# stages still run, fuzzy is just skipped.
_FUZZY_MAX_LINES = 20_000


class AmbiguousMatch(Exception):
    """The search block matches more than one location — never guess."""


@dataclass
class BlockMatch:
    start: int  # first matched line (inclusive)
    end: int  # past-the-end line
    window_indent: str = ""  # leading whitespace of the matched block
    search_indent: str = ""  # leading whitespace of the search block
    exact: bool = True


def _leading_ws(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _block_indent(lines: list[str]) -> str:
    """Shortest leading whitespace among non-empty lines."""
    indents = [_leading_ws(line) for line in lines if line.strip()]
    if not indents:
        return ""
    return min(indents, key=len)


def _unique_window(matches: list[int]) -> int:
    if len(matches) > 1:
        raise AmbiguousMatch()
    return matches[0] if matches else -1


def find_block(lines: list[str], search: list[str]) -> BlockMatch | None:
    """Locate ``search`` inside ``lines``, tolerantly. Raises AmbiguousMatch
    when several locations qualify at the same strictness level."""
    if not search or len(search) > len(lines):
        return None
    size = len(search)

    # 1. Exact line window (the caller usually checked str.count already, but
    #    line-based exactness also covers blocks not aligned on newlines).
    exact = [
        index for index in range(len(lines) - size + 1) if lines[index : index + size] == search
    ]
    at = _unique_window(exact)
    if at >= 0:
        return BlockMatch(at, at + size)

    # 2. Trailing-whitespace-insensitive.
    stripped_search = [line.rstrip() for line in search]
    relaxed = [
        index
        for index in range(len(lines) - size + 1)
        if [line.rstrip() for line in lines[index : index + size]] == stripped_search
    ]
    at = _unique_window(relaxed)
    if at >= 0:
        return BlockMatch(at, at + size, exact=False)

    # 3. Uniform indent shift: the model recopied the code at the wrong depth.
    search_indent = _block_indent(search)
    bare_search = [line[len(search_indent) :].rstrip() if line.strip() else "" for line in search]
    shifted: list[tuple[int, str]] = []
    for index in range(len(lines) - size + 1):
        window = lines[index : index + size]
        window_indent = _block_indent(window)
        bare_window = [
            line[len(window_indent) :].rstrip() if line.strip() else "" for line in window
        ]
        if bare_window == bare_search:
            shifted.append((index, window_indent))
    if len(shifted) > 1:
        raise AmbiguousMatch()
    if shifted:
        at, window_indent = shifted[0]
        return BlockMatch(
            at, at + size, window_indent=window_indent, search_indent=search_indent, exact=False
        )

    # 4. High-confidence fuzzy (difflib) — accepts one near-identical window.
    if len(lines) > _FUZZY_MAX_LINES:
        return None
    target = "\n".join(stripped_search)
    best_at, best, second = -1, 0.0, 0.0
    matcher = difflib.SequenceMatcher(autojunk=False)
    matcher.set_seq2(target)
    for index in range(len(lines) - size + 1):
        window = "\n".join(line.rstrip() for line in lines[index : index + size])
        matcher.set_seq1(window)
        ratio = matcher.ratio()
        if ratio > best:
            best_at, best, second = index, ratio, best
        elif ratio > second:
            second = ratio
    if best >= _FUZZY_ACCEPT and best - second >= _FUZZY_GAP:
        return BlockMatch(best_at, best_at + size, exact=False)
    return None


def closest_fragment(lines: list[str], search: list[str], max_lines: int = 20) -> str | None:
    """The file window most similar to the search block — given back to the
    model on failure so the NEXT attempt copies real text instead of guessing
    again. Returns None when nothing is even vaguely similar."""
    if not search or not lines:
        return None
    size = min(len(search), len(lines), max_lines)
    target = "\n".join(line.rstrip() for line in search[:size])
    matcher = difflib.SequenceMatcher(autojunk=False)
    matcher.set_seq2(target)
    best_at, best = -1, 0.0
    for index in range(len(lines) - size + 1):
        window = "\n".join(line.rstrip() for line in lines[index : index + size])
        matcher.set_seq1(window)
        ratio = matcher.ratio()
        if ratio > best:
            best_at, best = index, ratio
    if best < 0.5 or best_at < 0:
        return None
    return "\n".join(lines[best_at : best_at + size])


def reindent(replace: list[str], match: BlockMatch) -> list[str]:
    """Re-anchor the replacement at the matched block's real indentation when
    the match was found through an indent shift."""
    if match.window_indent == match.search_indent:
        return replace
    adjusted: list[str] = []
    for line in replace:
        if not line.strip():
            adjusted.append(line)
        elif line.startswith(match.search_indent):
            adjusted.append(match.window_indent + line[len(match.search_indent) :])
        else:
            adjusted.append(line)
    return adjusted
