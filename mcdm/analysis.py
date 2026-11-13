"""Évaluation absolue (MAUT), robustesse et comparaison des classements."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def maut(x, types, w, lo, hi) -> dict:
    """Score MAUT d'une alternative : S = Σ w_j u_j(x_j) avec utilités linéaires bornées.

    lo / hi : bornes de référence de chaque critère (fixées par le décideur, pas par l'échantillon).
    """
    x, lo, hi = (np.asarray(a, dtype=float) for a in (x, lo, hi))
    rng = np.where(hi - lo == 0, 1, hi - lo)
    u = np.where(np.array(types) == "max", (x - lo) / rng, (hi - x) / rng)
    u = np.clip(u, 0, 1)
    return {"utilities": u, "score": float(np.asarray(w) @ u)}


def spearman_matrix(rank_table: dict) -> tuple[np.ndarray, list]:
    names = list(rank_table)
    k = len(names)
    M = np.ones((k, k))
    for i in range(k):
        for j in range(k):
            if i != j:
                M[i, j] = spearmanr(rank_table[names[i]], rank_table[names[j]]).statistic
    return np.nan_to_num(M, nan=0.0), names


def spearman_critical(n: int) -> float:
    """Valeur critique (bilatérale, 5 %) de ρ de Spearman, approximation t pour n > 10."""
    table = {5: 1.0, 6: 0.886, 7: 0.786, 8: 0.738, 9: 0.700, 10: 0.648}
    if n in table:
        return table[n]
    from scipy.stats import t
    tc = t.ppf(0.975, n - 2)
    return float(tc / np.sqrt(n - 2 + tc ** 2))


def one_at_a_time(rank_fn, w, j, steps=21) -> tuple[np.ndarray, np.ndarray]:
    """Fait varier le poids du critère j de 0 à 1 ; les autres gardent leurs proportions."""
    w = np.asarray(w, dtype=float)
    others = np.delete(w, j)
    others = others / others.sum() if others.sum() > 0 else np.full(len(others), 1 / len(others))
    grid = np.linspace(0, 1, steps)
    ranks = []
    for g in grid:
        wn = np.insert(others * (1 - g), j, g)
        ranks.append(rank_fn(wn))
    return grid, np.array(ranks)


def monte_carlo(rank_fn, w, n_iter=500, concentration=50, seed=0) -> np.ndarray:
    """Tire des poids aléatoires autour de w (loi de Dirichlet) ; renvoie les rangs (n_iter, m)."""
    rng = np.random.default_rng(seed)
    alpha = np.maximum(np.asarray(w, dtype=float) * concentration, 1e-3)
    return np.array([rank_fn(rng.dirichlet(alpha)) for _ in range(n_iter)])
