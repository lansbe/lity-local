from __future__ import annotations

import re

# Catastrophic shell patterns, blocked UNCONDITIONALLY before execution — even
# in YOLO mode where the per-command confirm() is skipped. This is a heuristic
# CWE-78 denylist, NOT a sandbox: it stops the obvious foot-guns a model might
# emit (the real isolation layer, e.g. macOS sandbox-exec, is a separate step).
_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r":\s*\(\s*\)\s*\{.*:.*\|.*:.*\}", re.S), "fork bomb"),
    (
        re.compile(r"\b(?:curl|wget)\b[^|]*\|\s*(?:sudo\s+)?(?:sh|bash|zsh|python\d?)\b", re.I),
        "exécution d'un script distant (pipe vers un shell)",
    ),
    (re.compile(r"\bmkfs(?:\.\w+)?\b", re.I), "formatage d'un système de fichiers"),
    (re.compile(r"\bdd\b[^\n]*\bof=/dev/", re.I), "écriture brute sur un disque (dd of=/dev/…)"),
    (
        re.compile(r">\s*/dev/(?:sd|nvme|disk|hd)\w*", re.I),
        "redirection vers un périphérique disque",
    ),
    (re.compile(r"\bsudo\b", re.I), "élévation de privilèges (sudo) interdite à l'agent"),
    (re.compile(r"\b(?:shutdown|reboot|halt|poweroff)\b", re.I), "arrêt/redémarrage du système"),
    (
        re.compile(r"\bgit\b[^\n]*\bpush\b[^\n]*(?:--force\b|\s-f\b)", re.I),
        "git push forcé",
    ),
)

_RM_TARGETS = {"/", "~", "/*", "*", "$home", "$home/*", "~/*", "/.", "."}

# A bare, unexpanded shell variable as an rm -r target (`rm -rf $DIR`,
# `rm -r "$DIR"/*`): if the variable is empty or points at /, the command
# becomes `rm -rf /`. The agent can always use an explicit path instead.
_RM_BARE_VARIABLE = re.compile(r"\$\{?\w+\}?(?:/\*)?")


def _reckless_rm(text: str) -> bool:
    """A recursive `rm` aimed at the root, the home dir, a bare glob, or an
    unexpanded shell variable (whose value nobody verified)."""
    low = text.lower()
    if not re.search(r"\brm\b", low):
        return False
    after = low.split("rm", 1)[1]
    flags = re.findall(r"(?:^|\s)-(\w+)", after)
    if not any("r" in flag for flag in flags):
        return False
    tokens = re.split(r"\s+", after.replace("=", " ").strip())
    for token in tokens:
        bare = token.strip("\"'")
        if bare in _RM_TARGETS or _RM_BARE_VARIABLE.fullmatch(bare):
            return True
    return False


# Read-only inspection and project-check commands an autonomous agent may run
# WITHOUT a human in the loop (the YOLO allowlist). Anchored at the start of
# the command; any chaining/redirection/substitution disqualifies the whole
# line, because `pytest && rm -rf .` must never ride in on pytest's pass.
_AUTO_ALLOWED_PATTERNS = (
    r"git\s+(?:status|diff|log|show|branch|grep)\b",
    r"ls\b",
    r"pwd$",
    r"cat\b",
    r"head\b",
    r"tail\b",
    r"wc\b",
    r"grep\b",
    r"rg\b",
    r"find\b",
    r"tree\b",
    r"pytest\b",
    r"python3?\s+-m\s+(?:pytest|unittest|ruff)\b",
    r"python3?\s+--version$",
    r"uv\s+run\s+(?:pytest|ruff)\b",
    r"ruff\b",
    r"flake8\b",
    r"mypy\b",
    r"black\s+--check\b",
    r"npm\s+test\b",
    r"npm\s+run\s+(?:test|lint|build)\b",
    r"npx\s+tsc\b",
    r"node\s+--version$",
    r"cargo\s+(?:check|test|clippy)\b",
    r"go\s+(?:test|vet|build)\b",
    r"pip3?\s+(?:list|show)\b",
)
_AUTO_ALLOWED = tuple(re.compile(r"^" + pattern, re.I) for pattern in _AUTO_ALLOWED_PATTERNS)
_CHAINING = re.compile(r"[;&|<>`$]")


def is_auto_allowed(command: str, extra_patterns: tuple[str, ...] = ()) -> bool:
    """True when the command is safe to run unattended (YOLO allowlist).

    Conservative on purpose: chained, piped, redirected or substituted commands
    never qualify, even when they start with an allowlisted tool."""
    text = (command or "").strip()
    if not text or _CHAINING.search(text):
        return False
    if any(pattern.match(text) for pattern in _AUTO_ALLOWED):
        return True
    for raw in extra_patterns:
        try:
            if re.match(r"^" + raw, text, re.I):
                return True
        except re.error:
            continue
    return False


def is_dangerous(command: str) -> str | None:
    """Return a reason string if the command is catastrophic, else None."""
    text = (command or "").strip()
    if not text:
        return None
    if _reckless_rm(text):
        return "suppression récursive (rm -r) ciblant la racine / le home / un glob"
    for pattern, reason in _PATTERNS:
        if pattern.search(text):
            return reason
    return None
