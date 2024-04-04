import pytest
from math import isclose

import numpy as np
from scipy.stats import multivariate_normal, multivariate_t

import jams.helpers


def test_eval_inhomo1():
    a = np.identity(2)
    b = 2 * np.identity(2)
    assert isclose(jams.helpers.eval_inhomo(a, b), 1)

def test_eval_inhomo2():
    a = np.identity(2)
    b1 = np.array([(1, -1/2), (-1/2, 1)])
    b2 = np.array([(1, 1/2), (1/2, 1)])
    assert jams.helpers.eval_inhomo(a, b1) == jams.helpers.eval_inhomo(a, b2)

def test_eval_inhomo3():
    a = np.identity(2)
    b1 = np.array([(2, 0), (0, 1)])
    b2 = np.array([(4, 0), (0, 1)])
    assert jams.helpers.eval_inhomo(a, b1) < jams.helpers.eval_inhomo(a, b2)

def test_eval_inhomo4():
    a = np.identity(2)
    b1 = np.array([(1, 1/4), (1/4, 1)])
    b2 = np.array([(1, 1/2), (1/2, 1)])
    assert jams.helpers.eval_inhomo(a, b1) < jams.helpers.eval_inhomo(a, b2)

def test_seq_update_moments():
    m0 = np.zeros(2)
    c0 = np.identity(2)
    x = np.array([1, 2])
    m1, c1 = jams.helpers.seq_update_moments(x, 1, m0, c0)
    assert np.all(m1 == x)
    assert np.all(c1 == np.array([(1, 2), (2, 4)]))

def test_eval_cf_mvstud1():
    rng = np.random.default_rng()
    x = rng.standard_normal(2)
    m = rng.standard_normal(2)
    df = rng.exponential()
    c = np.outer(m, m) + np.outer(x, x)
    cf = np.linalg.cholesky(c)
    assert isclose(jams.helpers.eval_cf_mvstud(x, m, cf, df), multivariate_t.logpdf(x, m, c, df))

def test_eval_cf_mvstud2():
    rng = np.random.default_rng()
    x = rng.standard_normal(2)
    m = rng.standard_normal(2)
    df = np.inf
    c = np.outer(m, m) + np.outer(x, x)
    cf = np.linalg.cholesky(c)
    assert isclose(jams.helpers.eval_cf_mvstud(x, m, cf, df), multivariate_normal.logpdf(x, m, c, df))

def test_sample_cf_mvstud1():
    rng = np.random.default_rng(0)
    m = rng.standard_normal(2)
    mm = rng.standard_normal(2)
    df = np.inf
    c = np.outer(m, m) + np.outer(mm, mm)
    cf = np.linalg.cholesky(c)
    x = np.array([jams.helpers.sample_cf_mvstud(m, cf, df, rng) for _ in range(int(1e6))])
    k = (np.linalg.solve(cf, (x - m).T) ** 2).sum(0).mean()
    assert isclose(k, 2, rel_tol=1e-3)

def test_sample_cf_mvstud2():
    rng = np.random.default_rng(0)
    m = rng.standard_normal(2)
    mm = rng.standard_normal(2)
    df = 1e6
    c = np.outer(m, m) + np.outer(mm, mm)
    cf = np.linalg.cholesky(c)
    x = np.array([jams.helpers.sample_cf_mvstud(m, cf, df, rng) for _ in range(int(1e6))])
    k = (np.linalg.solve(cf, (x - m).T) ** 2).sum(0).mean()
    assert isclose(k, 2, rel_tol=1e-3)
