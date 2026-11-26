"""A.4: speculative decoding with a small draft model and a large target model that share a vocabulary.

The models are two callables. Each takes a list of token ids and returns a (len(ids), V) NumPy array whose row j is the
distribution of the token that follows position j. There is no KV cache, so this is the correctness reference; speed
is measured elsewhere (D.4).

Spec: openspec/specs/speculative-decoding/spec.md (while in flight: openspec/changes/conformal-and-speculative-decoding/)
"""

import collections
import itertools

import numpy as np
from scipy.stats import chi2


def _sample(p, rng):
    c = np.cumsum(p, dtype=np.float64)  # p need not sum to exactly 1
    return min(int(np.searchsorted(c, rng.random() * c[-1])), len(p) - 1)


def speculative_decode(target_probs, draft_probs, prefix, n_new, gamma, rng):
    """Return (ids, stats): the prefix plus exactly n_new new tokens, and the accept/reject counts.

    Each round drafts gamma tokens, scores them all with ONE target call and accepts each with probability
    min(1, p/q). At the first rejection the round stops and takes a token from the normalised (p - q)+. When every
    draft token is accepted the round adds one more token from the target. stats has target_calls, drafted, accepted,
    rejected and acceptance_rate = accepted / (accepted + rejected): draft tokens after a rejection are never judged,
    so they count as drafted only.
    """
    ids = list(prefix)
    stats = dict(target_calls=0, drafted=0, accepted=0, rejected=0)
    while len(ids) < len(prefix) + n_new:
        cur, qs, xs = list(ids), [], []
        for _ in range(gamma):
            q = draft_probs(cur)[-1]
            x = _sample(q, rng)
            qs.append(q)
            xs.append(x)
            cur.append(x)
        P = target_probs(cur)  # one target call scores every draft token and the position after the last one
        if P.shape[-1] != len(qs[0]):
            raise ValueError(f"draft vocabulary {len(qs[0])} differs from target vocabulary {P.shape[-1]}")
        stats["target_calls"] += 1
        stats["drafted"] += gamma
        n = len(ids)
        for i, x in enumerate(xs):
            p, q = P[n - 1 + i], qs[i]
            if rng.random() < min(1.0, p[x] / q[x]):
                ids.append(x)
                stats["accepted"] += 1
            else:
                r = np.maximum(p - q, 0.0)
                ids.append(_sample(r if r.sum() > 0 else p, rng))
                stats["rejected"] += 1
                break
        else:
            ids.append(_sample(P[-1], rng))  # every draft accepted: one free token
    judged = stats["accepted"] + stats["rejected"]
    stats["acceptance_rate"] = stats["accepted"] / judged if judged else float("nan")
    return ids[:len(prefix) + n_new], stats  # cutting the tail keeps the distribution


def _tables(V, seed):
    r = np.random.default_rng(seed)
    Tp = r.dirichlet(np.ones(V), size=V)  # target: row i = P(next token | last token i)
    Tq = 0.6 * Tp + 0.4 * r.dirichlet(np.ones(V), size=V)  # draft: related to the target but different
    return Tp, Tq


def _exact_dist(T, prefix, n, V):
    out = {}
    for seq in itertools.product(range(V), repeat=n):
        p, last = 1.0, prefix[-1]
        for t in seq:
            p *= T[last, t]
            last = t
        out[seq] = p
    return out


def _chi_test(counts, probs, N):
    seqs = list(probs)
    obs = np.array([counts.get(s, 0) for s in seqs], float)
    exp = np.array([probs[s] for s in seqs]) * N
    keep = exp >= 5
    stat = (((obs - exp) ** 2 / exp)[keep]).sum()
    pooled = bool((~keep).any())
    if pooled:
        stat += (obs[~keep].sum() - exp[~keep].sum()) ** 2 / exp[~keep].sum()
    return stat, chi2.sf(stat, int(keep.sum() + pooled - 1))


def _self_test():
    rng = np.random.default_rng(0)

    # identical draft and target: every draft token is accepted, a round adds gamma + 1 tokens with one target call
    Tp, Tq = _tables(5, 0)
    same = lambda ids: Tp[np.array(ids)]
    out, s = speculative_decode(same, same, [0], 5, 4, rng)
    assert len(out) == 6 and s["target_calls"] == 1 and s["accepted"] == 4 and s["rejected"] == 0 and s["drafted"] == 4
    assert s["acceptance_rate"] == 1.0
    # the surplus of the last round is cut: 3 requested, 5 produced
    out, _ = speculative_decode(same, same, [0], 3, 4, rng)
    assert len(out) == 4

    # a vocabulary mismatch is an error that names both sizes
    small, big = np.full((1, 5), 0.2), np.full((1, 6), 1 / 6)
    try:
        speculative_decode(lambda ids: np.tile(big, (len(ids), 1)), lambda ids: np.tile(small, (len(ids), 1)), [0], 1, 2, rng)
        raise AssertionError("a vocabulary mismatch was accepted")
    except ValueError as e:
        assert "5" in str(e) and "6" in str(e)

    # forced round: the draft always says token 0; the target accepts it after token 2, rejects it after token 0 and
    # then resamples token 1 from (p - q)+. One accepted, one rejected, the other two drafts only drafted.
    Tq3 = np.array([[1.0, 0, 0]] * 3)
    Tp3 = np.array([[0, 1.0, 0], [0, 1.0, 0], [1.0, 0, 0]])
    out, s = speculative_decode(lambda ids: Tp3[np.array(ids)], lambda ids: Tq3[np.array(ids)], [2], 2, 4, rng)
    assert out == [2, 0, 1], out
    assert (s["target_calls"], s["drafted"], s["accepted"], s["rejected"]) == (1, 4, 1, 1)
    assert s["acceptance_rate"] == 0.5

    # the output follows the target: 3-token continuations against the exact distribution; the draft alone fails
    V, n_new, gamma, N = 5, 3, 3, 60000
    tp, tq = (lambda ids: Tp[np.array(ids)]), (lambda ids: Tq[np.array(ids)])
    exact = _exact_dist(Tp, [0], n_new, V)
    cnt = collections.Counter(tuple(speculative_decode(tp, tq, [0], n_new, gamma, rng)[0][1:]) for _ in range(N))
    _, pv = _chi_test(cnt, exact, N)
    assert pv > 0.001, pv
    cnt_d = collections.Counter(tuple(speculative_decode(tq, tq, [0], n_new, gamma, rng)[0][1:]) for _ in range(N))
    _, pv_d = _chi_test(cnt_d, exact, N)
    assert pv_d < 1e-6, pv_d

    # acceptance rate of one-token rounds equals 1 - TV(p, q)
    acc = sum(speculative_decode(tp, tq, [0], 1, 1, rng)[1]["accepted"] for _ in range(N)) / N
    assert abs(acc - (1 - 0.5 * np.abs(Tp[0] - Tq[0]).sum())) < 0.01, acc
