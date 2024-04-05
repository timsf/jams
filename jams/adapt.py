import numpy as np
import numpy.typing as npt

import jams.helpers


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
        adapt_decay: float = .5,
        adapt_dilation: int = 1,
        adapt_smoother: float = 1e-3,
        adapt_target: float = .234,
    ):

        self.mean = self._running_mean = mean
        self.cov = self._running_cov = cov
        self.adapt_decay = adapt_decay
        self.adapt_dilation = adapt_dilation
        self.adapt_smoother = adapt_smoother
        self.adapt_target = self.emp_prob = adapt_target
        self.cf_cov = np.linalg.cholesky(self.cov)
        self.log_step_size = self._running_log_step_size = 0
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
            scale = np.exp(self.log_step_size * 2)
        else:
            scale = (2.38 ** 2 / len(state))
        return jams.helpers.sample_cf_mvstud(state, np.sqrt(scale) * self.cf_cov, df, rng)
    
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
            scale = np.exp(self.log_step_size * 2)
        else:
            scale = (2.38 ** 2 / len(state))
        return jams.helpers.eval_cf_mvstud(state, prop, np.sqrt(scale) * self.cf_cov, df)
    
    def eval_post_approx(self, val: FloatArr, df: float) -> float:
        """
        Evaluate the approximating t-density to the target.

        :param val: value at which to evaluate density
        :param df: degrees of freedom of t-distribution (set to np.inf for multivariate Gaussian)
        :return: log approximating density
        """
        
        return jams.helpers.eval_cf_mvstud(val, self.mean, self.cf_cov, df)

    def adapt(self, state: FloatArr, acc_prob: float):
        """
        Adjust the proposal mechanism for the last move of the Markov chain. Invoke this after every iteration.

        :param state: current position of the Markov chain
        :param acc_prob: acceptance probability of the current position at last accept-reject decision
        """

        self.iter += 1
        learning_rate = 1 / self.iter ** self.adapt_decay
        self.emp_prob[-1] = ((self.iter - self.epochs[-2] - 1) * self.emp_prob[-1] + acc_prob) / (self.iter - self.epochs[-2])
        # do not adapt mean
        _, self._running_cov = jams.helpers.seq_update_moments(state, 1 / learning_rate, self._running_mean, self._running_cov)
        self._running_log_step_size = self._running_log_step_size + (acc_prob - self.adapt_target) * learning_rate
        if self.iter == self.epochs[-1]:
            self.epochs.append(self.epochs[-1] + len(self.epochs) ** self.adapt_dilation)
            self.emp_prob.append(0.0)
            self.mean, self.cov = self._running_mean, self._running_cov
            self.log_step_size = self._running_log_step_size
            self.cf_cov = np.linalg.cholesky(self.cov + self.adapt_smoother * np.identity(len(state)))
