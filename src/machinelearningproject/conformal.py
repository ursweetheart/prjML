"""A.2: split conformal prediction for four-option questions, with the plain score 1 - p and randomized APS.

Spec: openspec/specs/conformal-prediction/spec.md (while in flight: openspec/changes/conformal-and-speculative-decoding/)
"""

import math

import numpy as np


def lac_scores(P, U=None):
    """Plain nonconformity score 1 - p for every option. P is (N, 4) probabilities; U is accepted and ignored."""
    return 1.0 - P


def aps_scores(P, U):
    """Randomized APS (Romano et al., 2020): probability mass of the options ranked above, plus U * p(option).

    P is (N, 4) probabilities and U is (N,) uniform numbers in [0, 1). Scores come back in the original option order.
    """
    order = np.argsort(-P, axis=1)  # most likely option first
    Ps = np.take_along_axis(P, order, axis=1)
    above = np.cumsum(Ps, axis=1) - Ps  # mass strictly above each option
    S = np.empty_like(P)
    np.put_along_axis(S, order, above + U[:, None] * Ps, axis=1)
    return S


def threshold(cal_scores, alpha):
    """The ceil((n+1)(1-alpha))-th smallest calibration score; infinite when that rank exceeds n (keep every option)."""
    n = len(cal_scores)
    k = math.ceil((n + 1) * (1 - alpha))
    return np.inf if k > n else np.sort(cal_scores)[k - 1]


def prediction_sets(scores, q):
    """(N, 4) boolean array: option j is in the set of question i when its score is at most q."""
    return scores <= q


def conformal(P, gold, cal, test, alpha, score_fn, rng):
    """Split conformal prediction.

    P: (N, 4) option probabilities in the original option order; gold: (N,) index of the right option;
    cal, test: boolean masks over the N questions; rng: a seeded numpy Generator (used by the adaptive score).
    Returns q_hat, the test sets (n_test, 4) and hit (n_test,), True where the right option is in its set.
    """
    nc, nt = int(cal.sum()), int(test.sum())
    s_cal = score_fn(P[cal], rng.random(nc))[np.arange(nc), gold[cal]]  # score of the TRUE option
    q = threshold(s_cal, alpha)
    sets = prediction_sets(score_fn(P[test], rng.random(nt)), q)
    hit = sets[np.arange(nt), gold[test]]
    return q, sets, hit


def coverage(hit):
    return float(np.mean(hit))


def mean_size(sets):
    return float(sets.sum(1).mean())


def empty_fraction(sets):
    return float((sets.sum(1) == 0).mean())


def _self_test():
    # scores of the worked example
    P = np.array([[0.5, 0.3, 0.15, 0.05]])
    assert np.allclose(lac_scores(P), [[0.5, 0.7, 0.85, 0.95]])
    assert np.allclose(aps_scores(P, np.zeros(1)), [[0.0, 0.5, 0.8, 0.95]])
    # APS is reported in the original option order even when the best option is not first
    assert np.allclose(aps_scores(np.array([[0.05, 0.15, 0.3, 0.5]]), np.zeros(1)), [[0.95, 0.8, 0.5, 0.0]])

    # threshold: the 9th of ten scores at alpha 0.2; infinite for 3 scores at alpha 0.1
    ten = np.array([0.05, 0.10, 0.15, 0.20, 0.30, 0.35, 0.50, 0.60, 0.70, 0.90])
    assert threshold(ten, 0.2) == 0.70
    assert threshold(ten[::-1], 0.2) == 0.70  # order of the scores does not matter
    assert threshold(np.array([0.1, 0.2, 0.3]), 0.1) == np.inf

    # the worked example keeps A and B (a margin of 1e-9 avoids the rounding of 1 - 0.3)
    assert prediction_sets(lac_scores(P), 0.70 + 1e-9).tolist() == [[True, True, False, False]]

    # same seed, same sets; the plain score does not depend on the seed
    rng0 = np.random.default_rng(0)
    Pr = rng0.dirichlet(np.ones(4), 400)
    gold = rng0.integers(0, 4, 400)
    cal = np.arange(400) < 200
    for fn in (lac_scores, aps_scores):
        a = conformal(Pr, gold, cal, ~cal, 0.1, fn, np.random.default_rng(7))
        b = conformal(Pr, gold, cal, ~cal, 0.1, fn, np.random.default_rng(7))
        assert a[0] == b[0] and np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2])
    plain1 = conformal(Pr, gold, cal, ~cal, 0.1, lac_scores, np.random.default_rng(1))
    plain2 = conformal(Pr, gold, cal, ~cal, 0.1, lac_scores, np.random.default_rng(2))
    assert np.array_equal(plain1[1], plain2[1])

    # an infinite threshold keeps every option: size 4, coverage 1, no empty set
    tiny = np.arange(400) < 3
    q, sets, hit = conformal(Pr, gold, tiny, ~tiny, 0.1, lac_scores, np.random.default_rng(0))
    assert q == np.inf and mean_size(sets) == 4 and coverage(hit) == 1 and empty_fraction(sets) == 0

    # simulated data: when the answers follow the probabilities, coverage is close to 1 - alpha (alpha 0.1)
    rng = np.random.default_rng(1)
    n = 20000
    Pf = rng.dirichlet(np.ones(4), n)
    gf = np.minimum((rng.random(n)[:, None] > np.cumsum(Pf, axis=1)).sum(1), 3)  # draw each answer from its row
    c = np.arange(n) < n // 2
    for fn in (lac_scores, aps_scores):
        cov = coverage(conformal(Pf, gf, c, ~c, 0.1, fn, rng)[2])
        assert 0.88 < cov < 0.92, (fn.__name__, cov)
