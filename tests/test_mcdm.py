"""Vérifie le moteur sur les exemples et exercices du cours. Lancer : python -m pytest -q"""
import numpy as np
from scipy.stats import spearmanr

from mcdm import weighting as W, ranking as R, fuzzy as F, analysis as A


def test_bwm_exemple_fournisseur():
    # Cours : Best = C1, Worst = C4, BO = (1,3,4,8), OW = (8,5,3,1), ξ* ≈ 0.08.
    # Le programme linéaire du cours a une solution unique (0.563, 0.216, 0.162, 0.060) ;
    # le tableau du cours (0.552, 0.207, 0.155, 0.086) ne vérifie pas |w1 − 8·w4| ≤ 0.08.
    res = W.bwm(0, 3, [1, 3, 4, 8], [8, 5, 3, 1])
    assert np.allclose(res["weights"], [0.563, 0.216, 0.162, 0.060], atol=0.002)
    assert abs(res["xi"] - 0.08) < 0.01
    assert list(np.argsort(-res["weights"])) == [0, 1, 2, 3]


def test_ahp_exercice_voiture():
    A_ = [[1, 7, 3], [1 / 7, 1, 1 / 3], [1 / 3, 3, 1]]
    res = W.ahp(A_)
    assert np.isclose(res["weights"].sum(), 1)
    assert res["weights"].argmax() == 0  # le coût domine
    assert res["consistent"]


def test_spearman_exemple_cours():
    topsis = [2, 1, 3, 5, 4, 6]
    vikor = [1, 2, 4, 3, 5, 6]
    prom = [2, 1, 3, 6, 4, 5]
    assert round(spearmanr(topsis, vikor).statistic, 2) == 0.77
    assert round(spearmanr(topsis, prom).statistic, 2) == 0.94
    assert round(spearmanr(vikor, prom).statistic, 2) == 0.60
    assert A.spearman_critical(6) == 0.886


def test_topsis_et_methodes_coherentes():
    X = [[7, 9, 9, 8], [8, 7, 8, 7], [9, 6, 8, 9], [6, 7, 8, 6]]
    t = ["max", "max", "max", "min"]
    w = [0.2, 0.1, 0.4, 0.3]
    for fn in (R.topsis, R.wsm, R.wpm, R.waspas, R.vikor, R.promethee_ii, R.electre_iii):
        res = fn(X, t, w)
        assert sorted(res["rank"]) == list(range(1, 5)) or len(set(res["rank"])) < 4


def test_entropie_critic_somme_un():
    X = np.random.default_rng(1).uniform(1, 10, (8, 5))
    t = ["max", "max", "min", "max", "min"]
    for fn in (W.entropy, W.critic):
        assert np.isclose(fn(X, t)["weights"].sum(), 1)


def test_fuzzy_topsis_dominance():
    # A0 domine A1 sur tous les critères → doit être premier
    Fm = np.array([[F.RATINGS["Très bon"], F.RATINGS["Bon"]],
                   [F.RATINGS["Moyen"], F.RATINGS["Mauvais"]]], dtype=float)
    Wf = np.array([F.IMPORTANCE["Forte"], F.IMPORTANCE["Moyenne"]])
    res = F.fuzzy_topsis(Fm, Wf, ["max", "max"])
    assert res["rank"][0] == 1


def test_maut_bornes():
    r = A.maut([10, 5000], ["max", "min"], [0.5, 0.5], [0, 5000], [10, 20000])
    assert np.isclose(r["score"], 1.0)
