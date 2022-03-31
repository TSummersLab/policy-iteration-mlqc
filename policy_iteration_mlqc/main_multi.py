import os
from time import time

import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt

from utility.pickle_io import pickle_export, pickle_import

from problem_data_gen import setup_rand_system
from experiment import experiment
from plotting import relative_plot


def load_and_plot(timestr):
    experiment_folder = './data'

    filename_in = 'monte_carlo_data_' + timestr + '.pkl'
    path_in = os.path.join(experiment_folder, filename_in)
    in_data = pickle_import(path_in)

    vi_num_iters_all = in_data['vi_num_iters_all']
    pi_num_iters_all = in_data['pi_num_iters_all']
    vi_time_elapsed_all = in_data['vi_time_elapsed_all']
    pi_time_elapsed_all = in_data['pi_time_elapsed_all']
    noise_levels_all = in_data['noise_levels_all']

    fig, axs = plt.subplots(ncols=2, figsize=(4, 3), sharey=True)

    relative_plot(noise_levels_all, vi_num_iters_all, pi_num_iters_all,
                  method1='vi', method2='pi', ydata_type='total iterations',
                  plot_type='scatter', pct_lwr=1, pct_upr=99,
                  fig=fig, ax=axs[0])

    relative_plot(noise_levels_all, vi_time_elapsed_all, pi_time_elapsed_all,
                  method1='vi', method2='pi', ydata_type='time elapsed',
                  plot_type='scatter', pct_lwr=1, pct_upr=99,
                  fig=fig, ax=axs[1])

    return fig, axs


if __name__ == "__main__":
    plt.close('all')

    run_experiments_from_scratch = False

    if run_experiments_from_scratch:
        num_trials = 1000
        rng = default_rng(seed=1)

        vi_num_iters_all, pi_num_iters_all = [], []
        vi_time_elapsed_all, pi_time_elapsed_all = [], []

        noise_levels_all = []

        for i in range(num_trials):
            noise_level = rng.uniform()
            system, K0, L0 = setup_rand_system(noise_level=noise_level, rng=rng)
            # system, K0, L0 = setup_pendulum_system(noise_level=noise_level)

            # Run the experiment
            vi_results, pi_results = experiment(system, K0, L0)

            vi_num_iters_all.append(vi_results['num_iters'])
            pi_num_iters_all.append(pi_results['num_iters'])

            vi_time_elapsed_all.append(vi_results['time_elapsed'])
            pi_time_elapsed_all.append(pi_results['time_elapsed'])

            noise_levels_all.append(noise_level)

            print('Trial %6d / %6d complete' % (i+1, num_trials))

        vi_num_iters_all, pi_num_iters_all = np.array(vi_num_iters_all), np.array(pi_num_iters_all)
        vi_time_elapsed_all, pi_time_elapsed_all = np.array(vi_time_elapsed_all), np.array(pi_time_elapsed_all)
        noise_levels_all = np.array(noise_levels_all)

        out_data = {'vi_num_iters_all': vi_num_iters_all,
                    'pi_num_iters_all': pi_num_iters_all,
                    'vi_time_elapsed_all': vi_time_elapsed_all,
                    'pi_time_elapsed_all': pi_time_elapsed_all,
                    'noise_levels_all': noise_levels_all}

        dirname_out = './data'
        timestr = str(time()).replace('.', 'p')
        filename_out = 'monte_carlo_data_' + timestr + '.pkl'
        pickle_export(dirname_out, filename_out, out_data)
    else:
        timestr = 'monte_carlo_data_1648682836p3982806'.split('_')[-1]

    load_and_plot(timestr=timestr)
