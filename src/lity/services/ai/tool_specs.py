from __future__ import annotations

import contextlib
from typing import Any


def tool_spec(
    name: str,
    description: str,
    properties: dict[str, tuple[str, str]],
    required: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    key: {"type": kind, "description": desc}
                    for key, (kind, desc) in properties.items()
                },
                "required": required or [],
            },
        },
    }


def build_tool_specs(
    *,
    allow_files: bool,
    allow_commands: bool,
    allow_write: bool,
    has_editor: bool,
    has_web: bool,
    retrieval: dict[str, Any],
    mcp: Any,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    if allow_files:
        specs += [
            tool_spec("list_files", "Liste les fichiers du répertoire de travail.", {}),
            tool_spec(
                "read_file",
                "Lit un fichier du répertoire de travail. Options pour lire une "
                "fenêtre : offset (1ʳᵉ ligne, 1-indexée) et limit (nombre de lignes).",
                {
                    "path": ("string", "Chemin relatif du fichier"),
                    "offset": ("integer", "Optionnel : première ligne à lire"),
                    "limit": ("integer", "Optionnel : nombre de lignes à lire"),
                },
                ["path"],
            ),
            tool_spec(
                "search",
                "Recherche une chaîne de texte dans les fichiers du répertoire.",
                {"query": ("string", "Texte à rechercher")},
                ["query"],
            ),
        ]
    if has_web:
        specs.append(
            tool_spec(
                "web_research",
                "Recherche web rapide et complète : cherche puis lit plusieurs sources "
                "en parallèle, et renvoie les passages pertinents avec leurs URL. "
                "À utiliser en priorité pour les questions récentes ou vérifiables.",
                {
                    "query": ("string", "Question ou requête de recherche précise"),
                    "max_sources": (
                        "integer",
                        "Optionnel : nombre de sources à lire en parallèle (1 à 4)",
                    ),
                },
                ["query"],
            )
        )
        specs.append(
            tool_spec(
                "web_search",
                "Recherche sur le web (actualités, faits récents, infos hors du projet). "
                "Renvoie des titres, URL et extraits.",
                {"query": ("string", "Requête de recherche web")},
                ["query"],
            )
        )
        specs.append(
            tool_spec(
                "fetch_url",
                "Ouvre une page web et renvoie son contenu texte nettoyé. Utilise une URL "
                "renvoyée par web_search.",
                {
                    "url": ("string", "URL http(s) de la page à lire"),
                    "query": (
                        "string",
                        "Optionnel : sujet recherché, pour extraire les passages pertinents",
                    ),
                },
                ["url"],
            )
        )
    if retrieval.get("project"):
        specs.append(
            tool_spec(
                "retrieve_project",
                "Recherche dans les documents/le code INDEXÉS du projet (RAG local). "
                "À utiliser pour répondre à propos du contenu du projet.",
                {"query": ("string", "Ce que tu cherches dans le projet")},
                ["query"],
            )
        )
    if retrieval.get("memory"):
        specs.append(
            tool_spec(
                "recall_memory",
                "Cherche dans tes conversations PASSÉES avec l'utilisateur (mémoire long "
                "terme). À utiliser pour te rappeler un fait/préférence déjà mentionné.",
                {"query": ("string", "Ce dont tu veux te souvenir")},
                ["query"],
            )
        )
    if allow_commands:
        specs.append(
            tool_spec(
                "run_command",
                "Exécute une commande shell dans le répertoire de travail et renvoie sa sortie.",
                {"command": ("string", "Commande shell à exécuter")},
                ["command"],
            )
        )
    if allow_write and has_editor:
        specs.append(
            tool_spec(
                "write_file",
                "Crée ou écrase un fichier du répertoire de travail avec le contenu fourni.",
                {
                    "path": ("string", "Chemin relatif du fichier"),
                    "content": ("string", "Contenu complet du fichier"),
                },
                ["path", "content"],
            )
        )
        specs.append(
            tool_spec(
                "edit_file",
                "Remplace un extrait exact dans un fichier existant du répertoire de travail.",
                {
                    "path": ("string", "Chemin relatif du fichier"),
                    "search": ("string", "Texte exact à remplacer"),
                    "replace": ("string", "Nouveau texte"),
                },
                ["path", "search", "replace"],
            )
        )
    if mcp is not None:
        with contextlib.suppress(Exception):  # pragma: no cover - defensive
            specs += list(mcp.tool_specs())
    return specs
