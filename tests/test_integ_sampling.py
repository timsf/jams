import pytest
from math import isclose

import numpy as np

import jams.sampling


def eval_logcomp(x, mu, sig):
    return -(len(x) * np.log(sig) + np.linalg.norm(x - mu, 2) ** 2 / sig) / 2

def test_acquisition():
    def eval_logp(x):
        return np.logaddexp(eval_logcomp(x, mu, sig1), eval_logcomp(x, -mu, sig2))
    def eval_d_logp(x):
        p = np.exp(eval_logp(x))
        dp = -((x - mu) / sig1 * np.exp(eval_logcomp(x, mu, sig1)) + (x + mu) / sig2 * np.exp(eval_logcomp(x, -mu, sig2)))
        return dp/p
    d = 2
    mu = 1
    sig1 = np.sqrt(d / 100)
    sig2 = sig1 / 2
    rng = np.random.default_rng(0)
    starting_points = rng.standard_normal(size=(32, d))
    ctrl = jams.sampling.Controls()
    modes, _ = jams.sampling.acquire_modes(eval_logp, eval_d_logp, starting_points, ctrl)
    assert len(modes) == 2
    assert np.all(np.isclose(mu, modes[0]))
    assert np.all(np.isclose(-mu, modes[1]))

def test_sampling():
    def eval_logp(x):
        return np.logaddexp(eval_logcomp(x, mu, sig1), eval_logcomp(x, -mu, sig2))
    def eval_d_logp(x):
        p = np.exp(eval_logp(x))
        dp = -((x - mu) / sig1 * np.exp(eval_logcomp(x, mu, sig1)) + (x + mu) / sig2 * np.exp(eval_logcomp(x, -mu, sig2)))
        return dp/p
    d = 2
    mu = 1
    sig1 = np.sqrt(d / 100)
    sig2 = sig1 / 2
    rng = np.random.default_rng(0)
    starting_points = rng.standard_normal(size=(32, d))
    sampler = jams.sampling.sample_posterior(eval_logp, eval_d_logp, starting_points, rng=rng)
    samples = [next(sampler) for _ in range(int(1e5))]
    x, i = (np.array(a) for a in zip(*samples[int(1e4):]))
    assert isclose(.5, np.mean(i), rel_tol=1e-2)
    assert np.all(np.isclose(mu, x[i == 0].mean(0), rtol=1e-2))
    assert np.all(np.isclose(-mu, x[i == 1].mean(0), rtol=1e-2))
