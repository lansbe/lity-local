---
name: revue-de-code
description: Relit du code pour trouver les bugs, les risques de sécurité et les améliorations de lisibilité, puis rend un avis structuré et priorisé. À utiliser quand l'utilisateur demande une revue de code, un avis sur du code, de repérer des bugs, ou de relire un fichier ou une fonction.
triggers: [revue, relis, relire, review, bug, bugs, sécurité, vulnérabilité, refactor, qualité, améliore]
---

# Revue de code

Quand on te demande de relire du code, rends une revue claire et actionnable.

## Méthode
1. Lis le code attentivement avant de juger. Si un fichier est mentionné mais pas chargé, demande-le ou inspecte-le avec les outils disponibles.
2. Cherche, dans cet ordre de priorité :
   - **Correction / bugs** : cas limites, off-by-one, `None`/null, exceptions non gérées, conditions inversées.
   - **Sécurité** : entrées non validées, injection, secrets en clair, chemins non contrôlés.
   - **Lisibilité / maintenabilité** : nommage, fonctions trop longues, duplication, complexité inutile.
   - **Performance** : seulement si un coût évident saute aux yeux.
3. Ne signale que de VRAIS problèmes. Pas de remarque cosmétique si le projet ne la suit pas déjà.

## Format de réponse
- Une ligne de verdict global.
- Une liste de constats — chacun : `gravité (bloquant / majeur / mineur)` · `fichier:ligne` · le problème · le correctif proposé (extrait court).
- Termine par les 1 à 3 actions prioritaires.

Reste concis : un constat = une vraie amélioration, pas un paragraphe.
