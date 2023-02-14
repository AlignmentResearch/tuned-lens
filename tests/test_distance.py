from torch.distributions import Categorical, Dirichlet, kl_divergence
from tuned_lens.stats import aitchison, aitchison_similarity, js_divergence, js_distance
import torch as th


def test_aitchison():
    # Uniform and symmetric Dirichlet prior over the simplex
    N, K = 100, 50_000
    prior = Dirichlet(th.ones(K, dtype=th.float64))

    log_x = prior.sample(th.Size([N])).log()
    log_y = prior.sample(th.Size([N])).log()
    log_z = prior.sample(th.Size([N])).log()
    weights = prior.sample(th.Size([N]))

    # Linearity:
    # <ax + by, z> = a<x, z> + b<y, z>
    a = th.randn(N, dtype=th.float64)
    b = th.randn(N, dtype=th.float64)

    th.testing.assert_close(
        # <ax + by, z>
        aitchison(
            th.log_softmax(a[:, None] * log_x + b[:, None] * log_y, dim=-1),
            log_z,
            weight=weights,
        ),
        # a<x, z> + b<y, z>
        (
            a * aitchison(log_x, log_z, weight=weights)
            + b * aitchison(log_y, log_z, weight=weights)
        ),
    )

    # Positive definiteness
    assert th.all(aitchison(log_x, log_x, weight=weights) >= 0)

    # Symmetry
    th.testing.assert_close(
        aitchison(log_x, log_y, weight=weights), aitchison(log_y, log_x, weight=weights)
    )

    # Cosine similarity properties
    self_similarities = aitchison_similarity(log_x, log_x, weight=weights)
    similarities = aitchison_similarity(log_x, log_y, weight=weights)
    assert th.all(similarities >= -1) and th.all(similarities <= 1)
    th.testing.assert_close(self_similarities, th.ones_like(self_similarities))


def test_js_divergence():
    p = Categorical(logits=th.randn(10))
    q = Categorical(logits=th.randn(10))
    m = Categorical(probs=0.5 * (p.probs + q.probs))  # type: ignore

    kl_fwd = kl_divergence(p, m)
    kl_bwd = kl_divergence(q, m)
    gt_js = 0.5 * (kl_fwd + kl_bwd)

    our_js_fwd = js_divergence(p.logits, q.logits)  # type: ignore
    our_js_bwd = js_divergence(q.logits, p.logits)  # type: ignore

    th.testing.assert_close(gt_js, our_js_fwd)
    th.testing.assert_close(our_js_fwd, our_js_bwd)  # Symmetry


def test_js_distance():
    a = th.randn(1000, 3)
    b = th.randn(1000, 3)
    c = th.randn(1000, 3)

    dist_ab = js_distance(a, b)
    dist_bc = js_distance(b, c)
    dist_ac = js_distance(a, c)

    # Triangle inequality
    assert th.all(dist_ab + dist_bc >= dist_ac)
