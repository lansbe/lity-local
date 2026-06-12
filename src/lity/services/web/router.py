from __future__ import annotations

import re

# A one-line, per-turn STEER appended to the agent system prompt (on top of the
# always-available web tools) when the user's question looks time-sensitive. It
# is purely ADDITIVE: it never hides web_search, so the model stays free to
# decide — this only reduces the "answered from stale memory" failure mode and
# cannot cause a false-negative lockout.
WEB_NUDGE = (
    "INDICE : cette question porte probablement sur des informations récentes ou "
    "qui évoluent dans le temps. Commence par web_research (puis fetch_url si un "
    "détail manque) avant de répondre, plutôt que de te fier à ta mémoire."
)

# Year tokens 2020-2099 + FR/EN markers of recency or fast-changing facts.
_YEAR_RE = re.compile(r"\b20[2-9]\d\b")
_TIME_MARKERS: tuple[str, ...] = (
    "aujourd'hui",
    "aujourdhui",
    "hier",
    "demain",
    "récent",
    "recent",
    "récente",
    "dernier",
    "dernière",
    "derniere",
    "actuel",
    "actuelle",
    "actuellement",
    "en ce moment",
    "ces jours",
    "cette année",
    "cette semaine",
    "ce mois-ci",
    "maintenant",
    "à jour",
    "prix",
    "tarif",
    "combien coûte",
    "combien coute",
    "cours de",
    "taux",
    "bourse",
    "météo",
    "meteo",
    "actualité",
    "actualités",
    "news",
    "nouvelle version",
    "dernière version",
    "derniere version",
    "vient de sortir",
    "sorti",
    "sortie",
    "classement",
    "score",
    "résultat",
    "resultat",
    "qui a gagné",
    "qui a gagne",
    "qui est le président",
)


def looks_time_sensitive(query: str) -> bool:
    """Heuristic: does this question likely need fresh/web info rather than memory?

    Used ONLY to add a steer toward web_search; web tools stay available either
    way, so a miss just means the model decides on its own (no lockout).
    Deliberately conservative to avoid nagging on evergreen questions.
    """
    if not query:
        return False
    lowered = query.lower()
    if _YEAR_RE.search(lowered):
        return True
    return any(marker in lowered for marker in _TIME_MARKERS)
