"""TOPSIS flou (Chen, 2000) avec nombres flous triangulaires — version du cours."""
from __future__ import annotations

import numpy as np

# Échelles linguistiques de Chen (2000)
RATINGS = {
    "Très mauvais": (0, 0, 1),
    "Mauvais": (0, 1, 3),
    "Plutôt mauvais": (1, 3, 5),
    "Moyen": (3, 5, 7),
    "Plutôt bon": (5, 7, 9),
    "Bon": (7, 9, 10),
    "Très bon": (9, 10, 10),
}
IMPORTANCE = {
    "Très faible": (0, 0, 0.1),
    "Faible": (0, 0.1, 0.3),
    "Plutôt faible": (0.1, 0.3, 0.5),
    "Moyenne": (0.3, 0.5, 0.7),
    "Plutôt forte": (0.5, 0.7, 0.9),
    "Forte": (0.7, 0.9, 1.0),
    "Très forte": (0.9, 1.0, 1.0),
}


def aggregate(tfns) -> np.ndarray:
    """Évaluation collective : moyenne des K nombres flous (étape 3 du cours)."""
    return np.mean(np.asarray(tfns, dtype=float), axis=0)


def defuzzify(tfn) -> float:
    """Centre de gravité (a+b+c)/3."""
    return float(np.mean(tfn, axis=-1))


def crisp_to_tfn(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.stack([x, x, x], axis=-1)


def fuzzy_topsis(F, W, types) -> dict:
    """F : matrice floue (m, n, 3) ; W : poids flous (n, 3) ; types : "max"/"min"."""
    F = np.asarray(F, dtype=float)
    W = np.asarray(W, dtype=float)
    m, n, _ = F.shape
    Rn = np.empty_like(F)
    for j, t in enumerate(types):
        if t == "max":
            c_star = F[:, j, 2].max() or 1
            Rn[:, j, :] = F[:, j, :] / c_star
        else:
            a_minus = F[:, j, 0].min()
            safe = np.where(F[:, j, :] == 0, 1e-9, F[:, j, :])
            Rn[:, j, :] = a_minus / safe[:, ::-1]  # (a⁻/c, a⁻/b, a⁻/a)
    V = Rn * W[None, :, :]
    d_plus = np.sqrt(((V - 1) ** 2).mean(axis=2)).sum(axis=1)
    d_minus = np.sqrt((V ** 2).mean(axis=2)).sum(axis=1)
    cc = d_minus / (d_plus + d_minus)
    from .ranking import ranks_from_scores
    return {"score": cc, "rank": ranks_from_scores(cc), "D_plus": d_plus,
            "D_minus": d_minus, "R": Rn, "V": V}
