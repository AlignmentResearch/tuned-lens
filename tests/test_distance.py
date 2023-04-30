import torch as th
from torch.distributions import Categorical, kl_divergence

from tuned_lens.stats import js_distance, js_divergence


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
