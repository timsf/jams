import pytest
from math import isclose

import numpy as np

import jams.adapt


def eval_logp(x):
    return -np.sum(np.square(x))/2

def update(x_nil, sampler, rng):
    x_prime = sampler.propose(x_nil, np.inf, rng)
    log_a = min(0, eval_logp(x_prime) - eval_logp(x_nil))
    if np.log(rng.uniform()) < log_a:
        return x_prime, np.exp(log_a)
    return x_nil, np.exp(log_a)

def test_adaptation():
    rng = np.random.default_rng(0)
    m = rng.standard_normal(2)
    x = rng.standard_normal(2)
    c = np.identity(2)
    sampler = jams.adapt.AdaptiveProposal(m, c * 2)
    for _ in range(int(1e6)):
        x, a = update(x, sampler, rng)
        sampler.adapt(x, a)
    assert isclose(np.mean(sampler.emp_prob[-1000:]), sampler.adapt_target, rel_tol=1e-3)
    assert np.all(np.isclose(c, sampler.cov, atol=1e-1))
