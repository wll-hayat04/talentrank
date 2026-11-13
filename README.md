# TalentRank — aide au recrutement par LLM + méthodes MCDM

Projet du cours *Aide à la décision* (EMI). Cas d'étude : recrutement d'un **Data Scientist junior**
parmi 12 candidats fictifs.

## Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

L'application s'ouvre avec des **scores de démonstration** : tout fonctionne sans clé API.
Pour faire noter les CV par un vrai LLM, choisis « LLM réel » dans la barre latérale :

- **Gratuit, en local** : installe [Ollama](https://ollama.com), puis `ollama pull llama3.1:8b`
  (aucune clé nécessaire).
- **API** : OpenAI, Groq, Mistral, ou tout service compatible OpenAI (URL + modèle + clé).

## Pipeline

| Étape | Onglet | Méthodes du cours |
|---|---|---|
| Formalisation (alternatives, critères, sens de préférence) | 1 | — |
| Extraction des scores depuis les CV | 2 | LLM extracteur sémantique (grille par critère, n passages, écart-type) |
| Évaluations linguistiques des recruteurs (C6) | 2 | Nombres flous triangulaires, agrégation, défuzzification |
| Pondération | 3 | AHP (λmax, CI, CR), BWM (ξ*), entropie, CRITIC, poids hybrides |
| Classement | 4 | WSM, WPM, WASPAS, TOPSIS, VIKOR (compromis C1/C2), PROMETHEE II, ELECTRE III, TOPSIS flou (Chen) |
| Évaluation absolue | 5 | MAUT, seuil fixé, seuil issu des données (ROC, Youden, distance au point idéal) |
| Robustesse et éthique | 6 | Spearman + valeur critique, sensibilité un-critère, Monte-Carlo, fiabilité LLM vs humains, test de biais sur le prénom |

## Structure

```
app.py              interface Streamlit (6 onglets)
llm.py              extraction des scores (API compatible OpenAI / Ollama) + test de biais
mcdm/weighting.py   AHP, BWM, entropie, CRITIC
mcdm/ranking.py     WSM, WPM, WASPAS, TOPSIS, VIKOR, PROMETHEE II, ELECTRE III
mcdm/fuzzy.py       échelles linguistiques, TOPSIS flou
mcdm/analysis.py    MAUT, Spearman, sensibilité, Monte-Carlo
data/demo.json      poste, critères, grilles, scores de démo, préférences des 3 recruteurs
data/cvs/*.txt      12 CV fictifs
tests/test_mcdm.py  vérifications sur les exemples du cours (python -m pytest -q)
```

## Remarques méthodologiques

- **Données fictives.** CV, scores de démo, avis des recruteurs et décisions passées sont inventés
  pour la démonstration. Ne jamais utiliser de vrais CV sans consentement.
- **BWM.** Sur l'exemple « fournisseur » du cours, le programme linéaire donne une solution unique
  (0,563 ; 0,216 ; 0,162 ; 0,060) avec ξ* = 0,084. Le tableau du cours (0,552 ; 0,207 ; 0,155 ; 0,086)
  ne respecte pas la contrainte |w1 − 8·w4| ≤ ξ : à signaler / vérifier avec l'enseignant.
- **PROMETHEE II / ELECTRE III.** Les seuils (q, p, v) s'appliquent aux valeurs normalisées sur [0, 1].
  ELECTRE III est classé par flux net de crédibilité (variante simplifiée de la distillation).
- **TOPSIS flou.** Critères issus du LLM convertis en nombres flous (x, x, x) ; C6 et les poids flous
  viennent des termes linguistiques des recruteurs.

## Mise à jour interface et fiabilité (novembre 2026)

- Nouvelle identité visuelle responsive, cartes et onglets améliorés, guide d'utilisation intégré.
- Gestion déterministe des ex æquo dans les classements.
- TOPSIS : proximité neutre de 0,5 si toutes les alternatives sont identiques.
- VIKOR : critère constant exclu de la distance à l'idéal.
- PROMETHEE II : support d'une seule alternative sans division par zéro.
- AHP : validation des matrices strictement positives.
- Extraction LLM : rejet des notes hors de l'échelle 0–10 et des salaires négatifs.
- Tests supplémentaires : `python -m pytest -q` (12 tests au total).

### Lancement

```bash
pip install -r requirements.txt
streamlit run app.py
```

Le mode démo fonctionne sans Ollama. Pour le mode LLM local, installer et lancer Ollama séparément puis télécharger le modèle configuré dans l'interface. Les sorties de modèles doivent être vérifiées avant toute utilisation dans une décision de recrutement.
