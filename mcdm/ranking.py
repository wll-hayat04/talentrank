"""Méthodes de classement des alternatives.

Additives / compensatoires : WSM, WPM, WASPAS, TOPSIS, VIKOR.
Surclassement (non compensatoires) : PROMETHEE II, ELECTRE III.
Toutes renvoient un dict avec "score" et "rank" (1 = meilleure alternative).
"""
from __future__ import annotations

import numpy as np

from .weighting import _linear_norm, _minmax_norm


def ranks_from_scores(scores, higher_is_better=True) -> np.ndarray:
    """Rang 1 = meilleur. Les ex aequo reçoivent le même rang (méthode 'min')."""
    s = np.asarray(scores, dtype=float)
    key = -s if higher_is_better else s
    if not np.isfinite(s).all():
        raise ValueError("Les scores du classement doivent être finis.")
    order = np.argsort(key, kind="stable")
    ranks = np.empty(len(s), dtype=int)
    if len(s) == 0:
        return ranks
    rank = 1
    ranks[order[0]] = rank
    for pos in range(1, len(s)):
        # Compare only adjacent sorted values; never merge distant near-ties transitively.
        if not np.isclose(key[order[pos]], key[order[pos - 1]], rtol=1e-10, atol=1e-12):
            rank = pos + 1
        ranks[order[pos]] = rank
    return ranks


# ----------------------------------------------------------------- WSM / WPM / WASPAS
def wsm(X, types, w):
    R = _linear_norm(X, types)
    q = R @ np.asarray(w)
    return {"score": q, "rank": ranks_from_scores(q), "R": R}


def wpm(X, types, w):
    R = _linear_norm(X, types)
    R = np.where(R <= 0, 1e-9, R)
    q = np.prod(R ** np.asarray(w), axis=1)
    return {"score": q, "rank": ranks_from_scores(q), "R": R}


def waspas(X, types, w, lam=0.5):
    q1 = wsm(X, types, w)["score"]
    q2 = wpm(X, types, w)["score"]
    q = lam * q1 + (1 - lam) * q2
    return {"score": q, "rank": ranks_from_scores(q), "Q_wsm": q1, "Q_wpm": q2}


# ------------------------------------------------------------------------- TOPSIS
def topsis(X, types, w):
    X = np.asarray(X, dtype=float)
    norm = np.sqrt((X ** 2).sum(axis=0))
    norm[norm == 0] = 1
    R = X / norm
    V = R * np.asarray(w)
    is_max = np.array([t == "max" for t in types])
    ideal = np.where(is_max, V.max(axis=0), V.min(axis=0))
    anti = np.where(is_max, V.min(axis=0), V.max(axis=0))
    s_plus = np.sqrt(((V - ideal) ** 2).sum(axis=1))
    s_minus = np.sqrt(((V - anti) ** 2).sum(axis=1))
    denom = s_plus + s_minus
    cc = np.divide(s_minus, denom, out=np.full_like(denom, 0.5), where=denom > 0)
    return {"score": cc, "rank": ranks_from_scores(cc), "S_plus": s_plus,
            "S_minus": s_minus, "V": V, "ideal": ideal, "anti_ideal": anti}


# -------------------------------------------------------------------------- VIKOR
def vikor(X, types, w, v=0.5):
    X = np.asarray(X, dtype=float)
    w = np.asarray(w)
    m = X.shape[0]
    is_max = np.array([t == "max" for t in types])
    best = np.where(is_max, X.max(axis=0), X.min(axis=0))
    worst = np.where(is_max, X.min(axis=0), X.max(axis=0))
    rng = best - worst
    rng[rng == 0] = 1
    G = w * (best - X) / rng
    G[:, np.isclose(best, worst)] = 0.0
    S, R = G.sum(axis=1), G.max(axis=1)
    dS = (S.max() - S.min()) or 1
    dR = (R.max() - R.min()) or 1
    Q = v * (S - S.min()) / dS + (1 - v) * (R - R.min()) / dR

    # Solution de compromis (conditions C1 et C2 du cours)
    order = np.argsort(Q, kind="stable")
    a1, a2 = order[0], order[1] if m > 1 else order[0]
    dq = 1 / (m - 1) if m > 1 else 0
    c1 = (Q[a2] - Q[a1]) >= dq
    c2 = (S[a1] == S.min()) or (R[a1] == R.min())
    if c1 and c2:
        compromise = [int(a1)]
    elif c1 and not c2:
        compromise = [int(a1), int(a2)]
    else:
        compromise = [int(i) for i in order if Q[i] - Q[a1] < dq]
    return {"score": Q, "rank": ranks_from_scores(Q, higher_is_better=False), "S": S, "R": R,
            "Q": Q, "C1_avantage": bool(c1), "C2_stabilite": bool(c2), "compromise": compromise}


# --------------------------------------------------------------------- PROMETHEE II
def _pref(d, kind, q, p):
    """Fonction de préférence appliquée à l'écart d (sur l'échelle normalisée [0,1])."""
    if kind == "usuelle":
        return (d > 0).astype(float)
    if kind == "en V":
        return np.clip(d / p, 0, 1) if p > 0 else (d > 0).astype(float)
    # linéaire avec zone d'indifférence (q, p)
    out = np.where(d <= q, 0.0, np.where(d >= p, 1.0, (d - q) / max(p - q, 1e-12)))
    return out


def promethee_ii(X, types, w, kind="linéaire", q=0.05, p=0.30):
    R = _minmax_norm(X, types)  # sens déjà pris en compte : plus grand = mieux
    w = np.asarray(w)
    m, n = R.shape
    Pi = np.zeros((m, m))
    for a in range(m):
        for b in range(m):
            if a != b:
                d = R[a] - R[b]
                Pi[a, b] = (w * _pref(d, kind, q, p)).sum()
    phi_plus = Pi.sum(axis=1) / max(m - 1, 1)
    phi_minus = Pi.sum(axis=0) / max(m - 1, 1)
    phi = phi_plus - phi_minus
    return {"score": phi, "rank": ranks_from_scores(phi), "phi_plus": phi_plus,
            "phi_minus": phi_minus, "Pi": Pi}


# ---------------------------------------------------------------------- ELECTRE III
def electre_iii(X, types, w, q=0.05, p=0.20, v=0.50):
    """ELECTRE III : indices de concordance, discordance et crédibilité σ(a,b).

    Seuils q (indifférence), p (préférence), v (veto) exprimés sur l'échelle normalisée [0,1].
    Classement final : flux net de crédibilité  Σ_b σ(a,b) − Σ_b σ(b,a)
    (variante simplifiée de la distillation, adaptée à une application interactive).
    """
    R = _minmax_norm(X, types)
    w = np.asarray(w)
    m, n = R.shape
    C = np.zeros((m, m))
    S = np.zeros((m, m))
    for a in range(m):
        for b in range(m):
            if a == b:
                continue
            d = R[b] - R[a]  # avantage de b sur a
            c_j = np.where(d <= q, 1.0, np.where(d >= p, 0.0, (p - d) / (p - q)))
            C[a, b] = (w * c_j).sum() / w.sum()
            D_j = np.where(d <= p, 0.0, np.where(d >= v, 1.0, (d - p) / (v - p)))
            sigma = C[a, b]
            for dj in D_j:
                if dj > C[a, b]:
                    sigma *= (1 - dj) / (1 - C[a, b]) if C[a, b] < 1 else 1
            S[a, b] = sigma
    net = S.sum(axis=1) - S.sum(axis=0)
    return {"score": net, "rank": ranks_from_scores(net), "concordance": C, "credibility": S}
