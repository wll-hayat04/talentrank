"""Extraction de scores à partir d'un CV avec un LLM (API compatible OpenAI).

Fonctionne avec OpenAI, Groq, Mistral, OpenRouter… et en local avec Ollama
(base_url = http://localhost:11434/v1, clé quelconque).
"""
from __future__ import annotations

import json
import re

import numpy as np

PROVIDERS = {
    "Ollama (local)": {"base_url": "http://localhost:11434/v1", "model": "llama3.1:8b"},
    "OpenAI": {"base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini"},
    "Groq": {"base_url": "https://api.groq.com/openai/v1", "model": "llama-3.3-70b-versatile"},
    "Mistral": {"base_url": "https://api.mistral.ai/v1", "model": "mistral-small-latest"},
    "Autre (compatible OpenAI)": {"base_url": "", "model": ""},
}

SYSTEM = (
    "Tu es un assistant de recrutement rigoureux et impartial. Tu notes un CV critère par critère "
    "en appliquant STRICTEMENT la grille fournie. Tu ignores le nom, le genre, l'âge, l'origine et la "
    "photo du candidat : seules les compétences et réalisations décrites comptent. "
    "Réponds uniquement en JSON valide."
)


def build_prompt(cv_text: str, job: dict, criteria: list[dict]) -> str:
    lines = [f"POSTE : {job['titre']}", f"DESCRIPTION : {job['description']}", "", "CRITÈRES ET GRILLES :"]
    for c in criteria:
        if c.get("numerique"):
            lines.append(f"- {c['code']} ({c['nom']}) : extraire le MONTANT numérique (MAD/mois). Si absent, null.")
        else:
            lines.append(f"- {c['code']} ({c['nom']}) : note entière de 0 à 10. Grille : {c['grille']}")
    keys = ", ".join(f'"{c["code"]}": {{"score": ..., "justification": "..."}}' for c in criteria)
    lines += ["", "CV :", '"""', cv_text.strip(), '"""', "",
              f"Format de réponse attendu (JSON, une justification courte en français par critère) : {{{keys}}}"]
    return "\n".join(lines)


def _parse(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("Réponse du LLM sans JSON.")
    return json.loads(m.group(0))


def _to_number(v):
    if v is None:
        return np.nan
    if isinstance(v, (int, float)):
        return float(v)
    digits = re.sub(r"[^\d.,]", "", str(v)).replace(",", ".")
    try:
        return float(digits.replace(" ", "")) if digits else np.nan
    except ValueError:
        return np.nan


def _chat(base_url: str, api_key: str, model: str, temperature: float, messages: list) -> str:
    """Appelle l'API /chat/completions (format OpenAI). Utilise le paquet openai s'il est installé,
    sinon requests (inclus avec Anaconda), pour éviter toute dépendance obligatoire."""
    try:
        from openai import OpenAI
    except ImportError:
        import requests

        url = (base_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
        r = requests.post(url, headers={"Authorization": f"Bearer {api_key or 'ollama'}"},
                          json={"model": model, "temperature": temperature, "messages": messages}, timeout=600)
        if r.status_code != 200:
            raise RuntimeError(f"Erreur {r.status_code} de l'API : {r.text[:200]}")
        return r.json()["choices"][0]["message"]["content"]
    client = OpenAI(base_url=base_url or None, api_key=api_key or "ollama")
    resp = client.chat.completions.create(model=model, temperature=temperature, messages=messages)
    return resp.choices[0].message.content


def score_cv(cv_text: str, job: dict, criteria: list[dict], *, base_url: str, api_key: str,
             model: str, n_runs: int = 3, temperature: float = 0.0) -> dict:
    """Note un CV n_runs fois. Renvoie {code: {"score": moyenne, "std": écart-type, "justification": str}}."""
    prompt = build_prompt(cv_text, job, criteria)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    runs = [_parse(_chat(base_url, api_key, model, temperature, messages)) for _ in range(n_runs)]
    out = {}
    for c in criteria:
        vals = [_to_number((r.get(c["code"]) or {}).get("score")) for r in runs]
        vals = np.array(vals, dtype=float)
        just = next(((r.get(c["code"]) or {}).get("justification", "") for r in runs if r.get(c["code"])), "")
        if not c.get("numerique"):
            vals[(vals < 0) | (vals > 10)] = np.nan
        else:
            vals[vals < 0] = np.nan
        out[c["code"]] = {"score": float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan,
                          "std": float(np.nanstd(vals)) if np.isfinite(vals).sum() > 1 else 0.0,
                          "justification": just}
    return out


def swap_name(cv_text: str, old_name: str, new_name: str) -> str:
    """Remplace le nom (et le prénom seul) pour le test de biais."""
    text = cv_text.replace(old_name, new_name)
    old_first, new_first = old_name.split()[0], new_name.split()[0]
    return re.sub(rf"\b{re.escape(old_first)}\b", new_first, text)
