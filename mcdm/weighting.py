"""Méthodes de pondération des critères (cours : Aide à la décision).

Subjectives : AHP, BWM.   Objectives : Entropie, CRITIC.
Convention : `types` est une liste de "max" (critère positif) ou "min" (critère négatif).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

# Indice de consistance aléatoire (table du cours, complétée pour n = 1, 2)
RI = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.56}


# --------------------------------------------------------------------------- AHP
def ahp(matrix) -> dict:
    """AHP par la méthode approximative du cours.

    1) somme des colonnes s ; 2) division de chaque élément par la somme de sa colonne ;
    3) poids = moyenne de chaque ligne ; 4) λmax = Σ s_i·w_i ; CI = (λmax−n)/(n−1) ; CR = CI/RI.
    """
    A = np.asarray(matrix, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or not np.isfinite(A).all() or np.any(A <= 0):
        raise ValueError("AHP : matrice carrée strictement positive attendue.")
    n = A.shape[0]
    s = A.sum(axis=0)
    norm = A / s
    w = norm.mean(axis=1)
    lam = float(s @ w)
    ci = (lam - n) / (n - 1) if n > 1 else 0.0
    ri = RI.get(n, 1.56)
    cr = ci / ri if ri > 0 else 0.0
    return {"weights": w, "lambda_max": lam, "CI": ci, "CR": cr,
            "consistent": cr < 0.1, "normalized": norm, "col_sums": s}


def ahp_matrix_from_upper(n: int, upper: dict) -> np.ndarray:
    """Construit la matrice réciproque à partir des valeurs a_ij (i<j)."""
    A = np.ones((n, n))
    for (i, j), v in upper.items():
        A[i, j] = v
        A[j, i] = 1.0 / v
    return A


# --------------------------------------------------------------------------- BWM
def bwm(best: int, worst: int, bo, ow) -> dict:
    """Best-Worst Method, forme linéaire du cours (min ξ).

    Contraintes : |w_B − a_Bj·w_j| ≤ ξ et |w_j − a_jW·w_W| ≤ ξ, Σw = 1, w ≥ 0.
    bo[j] = a_Bj (Best → autres), ow[j] = a_jW (autres → Worst).
    """
    bo = np.asarray(bo, dtype=float)
    ow = np.asarray(ow, dtype=float)
    n = len(bo)
    A_ub, b_ub = [], []
    for j in range(n):
        # w_B − a_Bj w_j − ξ ≤ 0   et   −w_B + a_Bj w_j − ξ ≤ 0
        r = np.zeros(n + 1); r[best] += 1; r[j] -= bo[j]; r[n] = -1
        A_ub.append(r); b_ub.append(0)
        r = np.zeros(n + 1); r[best] -= 1; r[j] += bo[j]; r[n] = -1
        A_ub.append(r); b_ub.append(0)
        # w_j − a_jW w_W − ξ ≤ 0   et   −w_j + a_jW w_W − ξ ≤ 0
        r = np.zeros(n + 1); r[j] += 1; r[worst] -= ow[j]; r[n] = -1
        A_ub.append(r); b_ub.append(0)
        r = np.zeros(n + 1); r[j] -= 1; r[worst] += ow[j]; r[n] = -1
        A_ub.append(r); b_ub.append(0)
    A_eq = [np.r_[np.ones(n), 0]]
    c = np.r_[np.zeros(n), 1]
    res = linprog(c, A_ub=np.array(A_ub), b_ub=b_ub, A_eq=np.array(A_eq), b_eq=[1],
                  bounds=[(0, None)] * (n + 1), method="highs")
    if not res.success:
        raise ValueError("BWM : le programme linéaire n'a pas de solution.")
    w = res.x[:n]
    return {"weights": w / w.sum(), "xi": float(res.x[n])}


# ------------------------------------------------------------------- normalisation
def _linear_norm(X, types):
    """Normalisation du cours : x/max pour les critères positifs, min/x pour les négatifs."""
    X = np.asarray(X, dtype=float)
    R = np.empty_like(X)
    for j, t in enumerate(types):
        col = X[:, j]
        if t == "max":
            R[:, j] = col / col.max() if col.max() != 0 else 0.0
        else:
            safe = np.where(col == 0, np.finfo(float).eps, col)
            R[:, j] = col.min() / safe
    return R


def _minmax_norm(X, types):
    X = np.asarray(X, dtype=float)
    R = np.empty_like(X)
    for j, t in enumerate(types):
        col = X[:, j]
        rng = col.max() - col.min()
        if rng == 0:
            R[:, j] = 1.0
        elif t == "max":
            R[:, j] = (col - col.min()) / rng
        else:
            R[:, j] = (col.max() - col) / rng
    return R


# ----------------------------------------------------------------------- Entropie
def entropy(X, types) -> dict:
    """Poids par l'entropie de Shannon : w_j = (1−E_j) / (n − ΣE_k)."""
    D = _linear_norm(X, types)
    m, n = D.shape
    P = D / D.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        plogp = np.where(P > 0, P * np.log(P), 0.0)
    E = -plogp.sum(axis=0) / np.log(m)
    d = 1 - E
    w = d / d.sum() if d.sum() > 0 else np.full(n, 1 / n)
    return {"weights": w, "E": E, "diversification": d}


# ------------------------------------------------------------------------- CRITIC
def critic(X, types) -> dict:
    """CRITIC : C_j = σ_j · Σ_k (1 − |ρ_jk|), w_j = C_j / ΣC."""
    R = _minmax_norm(X, types)
    sigma = R.std(axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = np.corrcoef(R, rowvar=False)
    rho = np.nan_to_num(rho, nan=0.0)
    np.fill_diagonal(rho, 1.0)
    C = sigma * (1 - np.abs(rho)).sum(axis=1)
    w = C / C.sum() if C.sum() > 0 else np.full(R.shape[1], 1 / R.shape[1])
    return {"weights": w, "sigma": sigma, "rho": rho, "C": C}
