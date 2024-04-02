import numpy as np
import numpy.typing as npt

from scipy.special import loggamma


FloatArr = npt.NDArray[np.float_]


class AdaptiveProposal(object):
    """
    Implements 'Adapting Increasingly Rarely (AIR)' adaptation mechanism for MCMC in Euclidean space. 
    Customized for JAMS by deactivating mean adaptation.

    :param mean: mean guess of target distribution
    :param cov: covariance guess of target distribution
    :param adapt_decay: rate at which the learning rate of the algorithm recedes. higher values stop adaptation more quickly
    :param adapt_dilation: rate at which adaptation windows grow. the proposal is adapted more rarely for higher values
    :param adapt_smoother: regularization parameter for the proposal covariance. higher values result in more smoothing
    :param adapt_target: targeted acceptance probability. higher values result in smaller step sizes
    """

    def __init__(
        self, 
        mean: FloatArr,
        cov: FloatArr,
        adapt_decay: float = .25,
        adapt_dilation: int = 1,
        adapt_smoother: float = 1e-4,
        adapt_target: float = .234,
    ):

        self.mean = self._running_mean = mean
        self.cov = self._running_cov = cov
        self.adapt_decay = adapt_decay
        self.adapt_dilation = adapt_dilation
        self.adapt_smoother = adapt_smoother
        self.adapt_target = adapt_target

        self.cf_prop_cov = np.linalg.cholesky(self.cov)
        self.log_step_size = [0]
        self.emp_prob = [0.0]
        self.epochs = [0, 1]
        self.iter = 0
        self.burnin = 0


    def propose(self, state: FloatArr, df: float, rng: np.random.Generator, est_step_size=True) -> FloatArr:
        """
        Propose a new state according to a multivariate t-distriution.

        :param state: current position of the Markov chain
        :param df: degrees of freedom of t-distribution (set to np.inf for multivariate Gaussian)
        :param rng: random state
        :param est_scale: whether to estimate optimal step size endogenously, or use benchmark value
        :return: random candidate value
        """
        
        if est_step_size:
            scale = np.exp(self.log_step_size[-1] * 2)
        else:
            scale = (2.38 ** 2 / len(state))
        return sample_cf_mvstud(state, np.sqrt(scale) * self.cf_prop_cov, df, rng)
    
    def eval(self, state: FloatArr, prop: FloatArr, df: float, est_step_size=True) -> float:
        """
        Evaluate the density of a candidate value under the proposal mechanism.

        :param state: current position of the Markov chain
        :param prop: candidate value
        :param df: degrees of freedom of t-distribution (set to np.inf for multivariate Gaussian)
        :param est_scale: whether to estimate optimal step size endogenously, or use benchmark value
        :return: log density of candidate value under the proposal mechanism
        """
        
        if est_step_size:
            scale = np.exp(self.log_step_size[-1] * 2)
        else:
            scale = (2.38 ** 2 / len(state))
        return eval_cf_mvstud(state, prop, np.sqrt(scale) * self.cf_prop_cov, df)
    
    def eval_post_approx(self, val: FloatArr, df: float) -> float:
        """
        Evaluate the approximating t-density to the target.

        :param state: value at which to evaluate density
        :param df: degrees of freedom of t-distribution (set to np.inf for multivariate Gaussian)
        :return: log approximating density
        """
        
        return eval_cf_mvstud(val, self.mean, self.cf_prop_cov, df)
    

    def adapt(self, state: FloatArr, acc_prob: float):
        """
        Adjust the proposal mechanism for the last move of the Markov chain. Invoke this after every iteration.

        :param state: current position of the Markov chain
        :param acc_prob: acceptance probability of the current position at last accept-reject decision
        """

        self.iter += 1
        self.emp_prob[-1] = ((self.iter - self.epochs[-2] - 1) * self.emp_prob[-1] + acc_prob) / (self.iter - self.epochs[-2])
        # do not adapt mean
        _, self._running_cov = seq_update_moments(state, 1 + len(self.emp_prob), self._running_mean, self._running_cov)
        if self.iter == self.epochs[-1]:
            learning_rate = 1 / (len(self.epochs) ** self.adapt_decay)
            self.log_step_size.append(self.log_step_size[-1] + learning_rate * (self.emp_prob[-1] - self.adapt_target))
            self.epochs.append(self.epochs[-1] + len(self.epochs) ** self.adapt_dilation)
            self.emp_prob.append(0.0)
            self.mean, self.cov = self._running_mean, self._running_cov
            self.cf_prop_cov = np.linalg.cholesky(self.cov + self.adapt_smoother * np.identity(len(state)))


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
        random_scale = 1 / rng.gamma(df / 2, df / 2)
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
        nc = loggamma((len(z) + df) / 2) - (loggamma(df / 2) + np.sum(np.log(np.diag(cf_cov))) - len(z) * (np.log(np.pi) + np.log(df)) / 2)
        kern = -(df + len(z)) * np.log(1 + np.sum(np.square(z)) / df) / 2
    return nc + kern
