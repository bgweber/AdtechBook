import numpy as np

from adtrain import selection as S


def test_recall_and_value_recall():
    V = np.array([[1.0, 3.0, 2.0], [5.0, 1.0, np.nan]])
    E = ~np.isnan(V)
    order = np.array([[2, 1, 0], [1, 0, 0]])  # second request: the winner is second
    r = S.recall(lambda k: order[:, :k], V, E)
    assert r[1]["recall"] == 0.0 and r[2]["recall"] == 1.0
    # k=1: the first request keeps 2 of 3, the second 1 of 5
    assert abs(r[1]["value_recall"] - (2 / 3 + 1 / 5) / 2) < 1e-3


def test_topk_respects_eligibility_and_merge_has_no_duplicates():
    E = np.array([[True, False, True, True]])
    Sp = np.array([[0.1, 9.0, 0.5, 0.3]])
    Sa = np.array([[0.9, 9.0, 0.1, 0.2]])
    assert list(S._topk(Sp, E, 2)[0]) == [2, 3]
    m = S.merged(Sp, Sa, E, 3)[0]
    assert len(set(m)) == 3 and 1 not in m and 0 in m  # affinity's pick fills a slot
