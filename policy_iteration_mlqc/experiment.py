import numpy as np
import numpy.random as npr
import numpy.linalg as la
import scipy.linalg as sla

from utility.matrixmath import vec, mat, mdot, matmul_lr, specrad, dlyap, dare, dare_gain
from quadtools import quadblock, quadstack, unquadblock, unquadstack


def experiment(system, K0, L0, show_diagnostic=False):
    system.K = K0
    system.L = L0
    # Check initial policy is stabilizing
    if specrad(system.linop1) > 1:
        raise ValueError('Initial policy is not stabilizing! '
                         'Check for stabilizability of the system or alter the policy gains.')

    X0 = system.policy_evaluation()

    # Baseline optimal value function and policies through value iteration w/ many iterations
    system.K, system.L = K0, L0
    vi_results_opt = system.value_iteration(num_iters=2000, show_diagnostic=False, save_hist=False)
    X_opt, K_opt, L_opt, time_elapsed_opt = [vi_results_opt[key] for key in ['X', 'K', 'L', 'time_elapsed']]
    P_opt, Phat_opt, S_opt, Shat_opt = unquadstack(X_opt)

    if show_diagnostic:
        print('Optimal')
        system.print_diagnostic(X_opt, K_opt, L_opt, X_opt)
        print('')


    # Value iteration
    if show_diagnostic:
        print('Value iteration')
    system.K, system.L = K0, L0
    vi_results = system.value_iteration(num_iters=2000, show_diagnostic=show_diagnostic, save_hist=True, X_opt=X_opt)
    X_vi, K_vi, L_vi, X_vi_hist, K_vi_hist, L_vi_hist, time_elapsed_vi = [vi_results[key] for key in ['X', 'K', 'L', 'X_hist', 'K_hist', 'L_hist', 'time_elapsed']]
    if show_diagnostic:
        print('')

    # Policy iteration
    if show_diagnostic:
        print('Policy iteration')
    system.K, system.L = K0, L0
    pi_results = system.policy_iteration(num_iters=20, show_diagnostic=show_diagnostic, save_hist=True, X_opt=X_opt)
    X_pi, K_pi, L_pi, X_pi_hist, K_pi_hist, L_pi_hist, time_elapsed_pi = [pi_results[key] for key in ['X', 'K', 'L', 'X_hist', 'K_hist', 'L_hist', 'time_elapsed']]
    if show_diagnostic:
        print('')

    def merit_base(X):
        return np.array([la.norm(X[i] - X_opt[i]) for i in range(4)])

    e0 = merit_base(X0)

    def merit(X):
        return np.max(merit_base(X)/e0)

    vi_results['e_hist'] = np.array([merit(X) for X in X_vi_hist])
    vi_results['num_iters'] = vi_results['e_hist'].size
    pi_results['e_hist'] = np.array([merit(X) for X in X_pi_hist])
    pi_results['num_iters'] = pi_results['e_hist'].size

    return vi_results, pi_results
