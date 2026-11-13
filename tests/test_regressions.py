import numpy as np
import pytest
from mcdm.ranking import ranks_from_scores, topsis, vikor, promethee_ii
from mcdm.weighting import ahp

def test_ties_stable_and_deterministic():
    assert ranks_from_scores([3, 3, 1]).tolist() == [1, 1, 3]
    assert ranks_from_scores([1, 2, 3], False).tolist() == [1, 2, 3]

def test_identical_candidates_topsis_are_neutral():
    r = topsis([[4, 5], [4, 5]], ["max", "min"], [.5, .5])
    assert np.allclose(r["score"], .5)
    assert r["rank"].tolist() == [1, 1]

def test_constant_criterion_does_not_affect_vikor():
    r = vikor([[3, 7], [8, 7]], ["max", "max"], [.5, .5])
    assert np.isfinite(r["score"]).all()
    assert r["rank"].tolist() == [2, 1]

def test_single_candidate_promethee():
    assert promethee_ii([[4]], ["max"], [1])["score"].tolist() == [0]

def test_invalid_ahp_matrix_rejected():
    with pytest.raises(ValueError):
        ahp([[1, 0], [1, 1]])
