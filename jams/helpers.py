import numpy as np
import numpy.typing as npt

from scipy.special import loggamma


FloatArr = npt.NDArray[np.float_]


def eval_inhomo(a: FloatArr, b: FloatArr) -> float:
    """
    Evaluate inhomogeneity coefficient between matrices a and b.

    :param a: benchmark matrix
    :param b: alternative matrix
    :return: inhomogeneity coefficent
    """

    c = np.linalg.solve(a, b).T
    eigvals, _ = np.linalg.eigh(c)
    return len(eigvals) * np.sum(1 / np.abs(eigvals)) / np.sum(1 / np.sqrt(np.abs(eigvals))) ** 2


def seq_update_moments(
    obs: FloatArr, 
    n: float, 
    mean: FloatArr, 
    cov: FloatArr,
) -> tuple[FloatArr, FloatArr]:
    
    """
    Sequentially update mean and covariance matrix estimates.

    :param obs: new observation to be absorbed into estimates
    :param n: number of samples that current estimate is based on
    :param mean: current mean estimate
    :param cov: current covariance estimate
    :return: updated mean and covariance estimates
    """

    dev = obs - mean
    mean = mean + dev / n
    cov = cov + (np.outer(dev, dev) - cov) / n
    return mean, cov


def sample_cf_mvstud(mean, cf_cov, df, rng):
    """
    Sample from a multivariate t-distribution, using the cholesky decomposition of its scale matrix.

    :param mean: location vector
    :param cov: lower cholesky factor of scale matrix
    :param df: degrees of freedom of t-distribution (set to np.inf for multivariate Gaussian)
    :param rng: random state
    :return: random sample from the distribution
    """

    if np.isinf(df):
        random_scale = 1
    else:
        random_scale = 1 / rng.gamma(df / 2, 2 / df)
    z = rng.standard_normal(size=len(mean))
    return mean + np.sqrt(random_scale) * cf_cov @ z


def eval_cf_mvstud(x, mean, cf_cov, df):
    """
    Evaluate density of multivariate t-distribution, using the cholesky decomposition of its scale matrix.

    :param x: position at which to evaluate density
    :param mean: location vector
    :param cov: lower cholesky factor of scale matrix
    :param df: degrees of freedom of t-distribution (set to np.inf for multivariate Gaussian)
    :return: log density
    """

    z = np.linalg.solve(cf_cov, x - mean)
    if np.isinf(df):
        nc = -(np.sum(np.log(np.diag(cf_cov))) + len(z) * np.log(2 * np.pi) / 2)
        kern = -np.sum(np.square(z)) / 2
    else:
        nc = loggamma((len(z) + df) / 2) - (loggamma(df / 2) + np.sum(np.log(np.diag(cf_cov))) + len(z) * (np.log(np.pi) + np.log(df)) / 2)
        kern = -(df + len(z)) * np.log(1 + np.sum(np.square(z)) / df) / 2
    return nc + kern
