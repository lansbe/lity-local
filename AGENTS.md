# AGENTS.md

## Projet

Lity est une application desktop Python locale. Elle combine :

- chat IA via Ollama, LM Studio et fournisseurs CLI configurés localement;
- mémoire locale JSON et conversations multiples;
- personnages locaux optionnels avec instructions par conversation et packs d'émotions générés;
- contexte de fichiers, RAG projet et propositions de modifications validées par l'interface;
- compétences locales au format `SKILL.md`;
- génération locale d'images et de vidéos quand les runtimes optionnels sont installés;
- STT/TTS local;
- interface web desktop pywebview, interface Qt et mode console.

## Architecture

Le code applicatif vit sous `src/lity/`.

- `app/` contient le contrôleur, les mixins et les entrypoints.
- `core/` contient les fonctions pures et modèles simples.
- `infrastructure/` contient les chemins applicatifs, settings et logging.
- `interfaces/` contient les adapters utilisateur : CLI, desktop Qt et desktop web.
  - `desktop_web/` expose `DesktopApi` (pont pywebview sérialisable JSON) et `run_desktop_web`.
  - Le frontend React vit dans `frontend/` et se construit dans `desktop_web/web_dist/`.
- `services/` contient les intégrations et sous-systèmes.
  - `services/memory/` gère mémoire long terme et conversations multiples.
  - `services/characters/` gère les personnages créés par l'utilisateur, leurs profils JSON et leurs images d'émotions.
  - `services/ai/agent.py` fournit la boucle agent à outils.
  - `services/rag/` fournit l'indexation et la récupération.
  - `services/skills/` charge les compétences intégrées et utilisateur.
  - `services/audio/`, `image_generation/` et `video_generation/` gardent les intégrations optionnelles isolées.
- `resources/` contient les ressources packagées et compétences intégrées.

Évite de recréer des modules racine. Les nouveaux modules doivent aller dans le package `lity`.
Le cœur applicatif ne doit dépendre ni de PySide6 ni de pywebview ; `DesktopApi` reste testable sans fenêtre grâce à ses callbacks injectés.

## Règles De Développement

- Garde les fichiers courts et focalisés. Si un fichier dépasse environ 300 lignes, cherche une séparation naturelle.
- Ne mets pas de dépendance UI dans le cœur applicatif.
- Ne fais pas d'appel réel à Ollama, audio, web search, Stable Diffusion ou génération vidéo dans les tests unitaires.
- Utilise `AppPaths` pour les chemins applicatifs ; n'écris pas directement dans un dossier de données depuis du nouveau code.
- Les imports lourds ou optionnels doivent rester paresseux quand possible.
- Les écritures de fichiers utilisateur doivent passer par `FileManager`, `CodeEditor`, `SettingsStore` ou `MemoryManager`.
- Les profils et images de personnages doivent rester sous `AppPaths.characters_dir`.
- Les modifications de code proposées par l'IA doivent rester validées par l'interface avant écriture.

## Commandes

```bash
uv sync --extra desktop --extra web --extra dev --extra packaging
uv lock
uv run lity --ui web        # UI web (frontend build requis)
uv run lity --ui web --dev  # UI web sur le serveur Vite
uv run lity --ui qt         # UI PySide6
uv run lity --console
uv run pytest
uv run ruff check .
uv run ruff format .
```

Frontend web :

```bash
cd frontend
npm install
npm run dev      # développement
npm run build    # -> src/lity/interfaces/desktop_web/web_dist
```

## Packaging

PyInstaller est la première cible pratique pour macOS et Windows. Il faut construire l'app sur l'OS cible.

```bash
uv sync --extra desktop --extra web --extra packaging
uv run pyinstaller packaging/pyinstaller/lity.spec --noconfirm
```

Scripts disponibles :

```bash
./scripts/build_macos.sh
```

```powershell
.\scripts\build_windows.ps1
```

## Tests

Les tests doivent être déterministes, avec `tmp_path` ou `tempfile`, et ne doivent pas dépendre d'un serveur local.

Si tu ajoutes une intégration externe, crée une interface ou un faux objet injecté dans les tests.
