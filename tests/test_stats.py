import random

import torch as th
from torch.distributions import Dirichlet, kl_divergence

from tuned_lens.stats import LogitStats


def test_logit_stats_correctness():
    """Test that `LogitStats` recovers the true Dirichlet within a small error."""
    th.manual_seed(42)

    x = Dirichlet(th.tensor([1.0, 1.0, 1.0]))
    logits1 = x.sample(th.Size([10000])).log() + random.uniform(-0.1, 0.1)
    logits2 = x.sample(th.Size([10000])).log() + random.uniform(-0.1, 0.1)

    stats = LogitStats()
    stats.update(logits1)
    stats.update(logits2)
    x2 = stats.mle()

    assert kl_divergence(x, x2) < 1e-3
