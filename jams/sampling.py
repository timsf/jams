from typing import Callable, Iterator, NamedTuple

import numpy as np
import numpy.typing as npt
import networkx as nw
import scipy.optimize
import scipy.spatial.distance
import scipy.special

import jams.adapt, jams.helpers


FloatArr = npt.NDArray[np.float_]


class Controls(NamedTuple):
    """
    Tuning parameter collection.

    :param acq_thresh: ['epsilon'] distance threshold below which similar mode candidates are merged during mode acquisition
    :param adapt_decay ['alpha']: rate at which the learning rate of the algorithm recedes. higher values stop adaptation more quickly
    :param adapt_dilation: rate at which adaptation windows grow. the proposal is adapted more rarely for higher values
    :param adapt_min_batch: minimum adaptation window to be reached before terminating burn-in algorithm
    :param adapt_smoother: ['beta'] regularization parameter for the proposal covariance. higher values result in more smoothing
    :param adapt_target: targeted acceptance probability. higher values result in smaller step sizes
    :param adapt_thresh: ['b_acc'] burn-in algorithm termination criterion. values closer to 1 result in longer burn-in time, larger values in shorter burn-in time
    :param jump_prob: probability of attempting a jump move at each iteration
    :param jump_kern_df: degrees of freedom of components in augmented model
    :param jump_weight_lb: ['epsilon_w'] lower bound of component weight in augmented model
    :param prop_kern_df: degrees of freedom of proposal distribution
    """

    acq_thresh: float = .1
    adapt_decay: float = .66
    adapt_dilation: float = 1
    adapt_min_batch: int = 100
    adapt_smoother: float = 1e-4
    adapt_target: float = .234
    adapt_thresh: float = 1.01
    jump_prob: float = .1
    jump_kern_df: float = 7
    jump_weight_lb: float = 1e-2
    prop_kern_df: float = np.inf


def sample_posterior(
    eval_logp: Callable[[FloatArr], float],
    eval_d_logp: Callable[[FloatArr], FloatArr],
    starting_points: FloatArr,
    ctrl: Controls | None = Controls(),
    rng: np.random.Generator = np.random.default_rng(0),
) -> Iterator[tuple[FloatArr, float]]:
    
    """
    Sample from augmented target distribution.

    :param eval_logp: ['pi'] log target density 
    :param eval_d_logp: gradient of log target density
    :param starting_points: initial values for mode acquisition
    :param ctrl: tuning parameter collection
    :param rng: random state
    :return: generator that yields an iteration from the MCMC algorithm at each invocation 
    """
    
    samplers = warm_up(eval_logp, eval_d_logp, starting_points, ctrl, rng)
    i = rng.choice(len(samplers))
    x = samplers[i].mean
    while True:
        if rng.uniform() < ctrl.jump_prob:
            x, i = update_globally(x, i, eval_logp, eval_d_logp, samplers, ctrl, rng)
        else:
            x, a = update_locally(x, i, eval_logp, eval_d_logp, samplers, ctrl, rng)
            samplers[i].adapt(x, a)
        yield x, i


def update_locally(
    x_nil: FloatArr,
    i: int,
    eval_logp: Callable[[FloatArr], float],
    eval_d_logp: Callable[[FloatArr], FloatArr],
    samplers: list[jams.adapt.AdaptiveProposal],
    ctrl: Controls,
    rng: np.random.Generator,
) -> tuple[FloatArr, float]:
    """
    Attempt a move within-mode.

    :param x_nil: main state variable
    :param i: auxiliary variable
    :param eval_logp: ['pi'] log target density 
    :param eval_d_logp: gradient of log target density
    :param samplers: adaptive local approximation for each mode
    :param ctrl: tuning parameter collection
    :param rng: random state
    :return: update to main state variable, acceptance probability of proposed move
    """
    
    x_prime = samplers[i].propose(x_nil, ctrl.prop_kern_df, rng)
    log_joint_nil = eval_joint(x_nil, i, eval_logp, eval_d_logp, samplers, ctrl)
    log_joint_prime = eval_joint(x_prime, i, eval_logp, eval_d_logp, samplers, ctrl)
    log_acc_prob = min(0, log_joint_prime - log_joint_nil)
    if np.log(rng.uniform()) < log_acc_prob:
        return x_prime, np.exp(log_acc_prob)
    return x_nil, np.exp(log_acc_prob)


def update_globally(
    x_nil: FloatArr,
    i_nil: int,
    eval_logp: Callable[[FloatArr], float],
    eval_d_logp: Callable[[FloatArr], FloatArr],
    samplers: list[jams.adapt.AdaptiveProposal],
    ctrl: Controls,
    rng: np.random.Generator,
) -> tuple[FloatArr, int]:
    """
    Attempt a move between modes, using the 'corresponding point' scheme.

    :param x_nil: main state variable
    :param i_nil: auxiliary variable
    :param eval_logp: ['pi'] log target density 
    :param eval_d_logp: gradient of log target density
    :param samplers: adaptive local approximation for each mode
    :param ctrl: tuning parameter collection
    :param rng: random state
    :return: update to main state and auxiliary variables
    """

    i_prime = rng.choice([i for i in range(len(samplers)) if i != i_nil])
    x_prime = samplers[i_prime].mean + samplers[i_prime].cf_cov @ np.linalg.solve(samplers[i_nil].cf_cov, x_nil - samplers[i_nil].mean)
    log_joint_nil = eval_joint(x_nil, i_nil, eval_logp, eval_d_logp, samplers, ctrl)
    log_joint_prime = eval_joint(x_prime, i_prime, eval_logp, eval_d_logp, samplers, ctrl)
    # change of variable adjustments correspond to diagonal of cholesky factors
    log_jac_nil = np.sum(np.log(np.diag(samplers[i_nil].cf_cov)))
    log_jac_prime = np.sum(np.log(np.diag(samplers[i_prime].cf_cov))) 
    log_acc_prob = min(0, log_joint_prime - log_joint_nil + log_jac_prime - log_jac_nil)
    if np.log(rng.uniform()) < log_acc_prob:
        return x_prime, i_prime
    return x_nil, i_nil


def eval_joint(
    x: FloatArr,
    i: int,
    eval_logp: Callable[[FloatArr], float],
    eval_d_logp: Callable[[FloatArr], FloatArr],
    samplers: list[jams.adapt.AdaptiveProposal],
    ctrl: Controls,
) -> float:
    """
    Evaluate the augmented target density ['pi_tilde'].

    :param x: main state variable
    :param i: auxiliary variable
    :param eval_logp: ['pi'] log target density 
    :param eval_d_logp: gradient of log target density
    :param samplers: adaptive local approximation for each mode
    :param ctrl: tuning parameter collection
    :return: log augmented target density
    """

    log_p = eval_logp(x)
    log_q = np.array([sampler.eval_post_approx(x, ctrl.jump_kern_df) for sampler in samplers])
    n_samples = np.array([sampler.iter - sampler.burnin + 1 for sampler in samplers])
    log_w = np.log(n_samples + np.sum(n_samples) / (len(samplers) * (1 / ctrl.jump_weight_lb - 1)))
    log_joint = log_p + log_q[i] + log_w[i] - scipy.special.logsumexp(log_w + log_q)
    return log_joint


def warm_up(
    eval_logp: Callable[[FloatArr], float],
    eval_d_logp: Callable[[FloatArr], FloatArr],
    starting_points: FloatArr,
    ctrl: Controls,
    rng: np.random.Generator,
) -> list[jams.adapt.AdaptiveProposal]:
    """
    Acquire the modes of the target density, and initialize the local approximations at each of the modes.

    :param eval_logp: ['pi'] log target density 
    :param eval_d_logp: gradient of log target density
    :param starting_points: initial values for mode acquisition
    :param ctrl: tuning parameter collection
    :param rng: random state
    :return: local approximation at each mode
    """
    
    modes, covs = acquire_modes(eval_logp, eval_d_logp, starting_points, ctrl)
    samplers = [jams.adapt.AdaptiveProposal(mode, cov, ctrl.adapt_decay, ctrl.adapt_dilation, ctrl.adapt_smoother, ctrl.adapt_target) 
                for mode, cov in zip(modes, covs)]
    
    x = [sampler._running_mean for sampler in samplers]
    inhomo = [np.inf for _ in samplers]
    lagged_cov = [sampler.cov for sampler in samplers]
    while True:
        for i in range(len(samplers)):
            x[i], a = update_locally(x[i], i, eval_logp, eval_d_logp, samplers, ctrl, rng)
            samplers[i].adapt(x[i], a)
            samplers[i].burnin = samplers[i].iter
            if samplers[i].epochs[-2] == samplers[i].iter:
                # check that at least one move has been accepted
                if np.any(x[i] != samplers[i].mean) and samplers[i].epochs[-1] - samplers[i].epochs[-2] > ctrl.adapt_min_batch:
                    inhomo[i] = jams.helpers.eval_inhomo(samplers[i].cov, lagged_cov[i])
                lagged_cov[i] = samplers[i].cov
        if max(inhomo) < ctrl.adapt_thresh:
            return samplers


def acquire_modes(
    eval_logp: Callable[[FloatArr], float],
    eval_d_logp: Callable[[FloatArr], FloatArr],
    starting_points: FloatArr,
    ctrl: Controls, 
) -> tuple[list[FloatArr], list[FloatArr]]:
    """
    Acquire the modes of the target density.

    :param eval_logp: ['pi'] log target density 
    :param eval_d_logp: gradient of log target density
    :param starting_points: initial values for mode acquisition
    :param ctrl: tuning parameter collection
    :return: list of modes and corresponding inverse (possibly approximate) Hessians
    """
    
    eval_rec_logp = lambda x: -eval_logp(x)
    eval_rec_d_logp = lambda x: -eval_d_logp(x)
    minimizers = [scipy.optimize.minimize(eval_rec_logp, jac=eval_rec_d_logp, x0=x0, method='BFGS') for x0 in starting_points]
    # consolidate adjacent local maxima
    dists = np.array([[(m1.x - m2.x) @ (m1.hess_inv + m2.hess_inv) @ (m1.x - m2.x) for m1 in minimizers] for m2 in minimizers])
    comps = list(nw.connected_components(nw.Graph(dists < ctrl.acq_thresh)))
    modes = [np.mean([minimizers[i].x for i in comp], 0) for comp in comps]
    covs = [np.mean([minimizers[i].hess_inv for i in comp], 0) for comp in comps]
    return modes, covs
