import numpy as np
import matplotlib.pyplot as plt

from problem_data_gen import setup_pendulum_system
from experiment import experiment
from plotting import single_system_plot


if __name__ == "__main__":
    plt.close('all')

    vi_results_all, pi_results_all = [], []
    noise_levels = np.array([0.00, 0.1, 1.00])

    system, K0, L0 = setup_pendulum_system()

    a_orig, b_orig, c_orig = np.copy(system.a), np.copy(system.b), np.copy(system.c)

    for noise_level in noise_levels:
        # Set up the problem
        system.a = noise_level*a_orig
        system.b = noise_level*b_orig
        system.c = noise_level*c_orig

        # Run the experiment
        vi_results, pi_results = experiment(system, K0, L0, show_diagnostic=True)

        # Store the results
        vi_results_all.append(vi_results)
        pi_results_all.append(pi_results)

    # Plots
    single_system_plot(vi_results_all, pi_results_all, noise_levels, x_axis_type='Iterations')
    single_system_plot(vi_results_all, pi_results_all, noise_levels, x_axis_type='Wall clock time (s)')
