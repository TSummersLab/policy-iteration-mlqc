import math

import numpy as np
import numpy.random as npr
import numpy.linalg as la
import scipy.linalg as sla

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle


def get_clipped_cmap(name, low=0.0, high=1.0):
    cmap_orig = get_cmap(name, 512)
    return ListedColormap(cmap_orig(np.linspace(low, high, 256)))


def single_system_plot(vi_results_all, pi_results_all, noise_levels, x_axis_type='Iterations'):
    fig, ax = plt.subplots(figsize=(6, 4))

    vi_cmap = get_clipped_cmap('Reds', low=0.3, high=0.9)
    pi_cmap = get_clipped_cmap('Blues', low=0.3, high=0.9)

    num_levels = len(noise_levels)

    def plot_one(results_all, cmap, linestyle, method_str):
        for i, (results, noise_level) in enumerate(zip(results_all, noise_levels)):
            y_axis_values = results['e_hist']
            num_iters = results['num_iters']
            avg_clock_time_hist = results['time_elapsed']*np.linspace(0, 1, num_iters)

            if x_axis_type == 'Iterations':
                x_axis_values = np.arange(num_iters)
            elif x_axis_type == 'Wall clock time (s)':
                x_axis_values = avg_clock_time_hist

            plt.semilogy(x_axis_values, y_axis_values, c=cmap(i/(num_levels - 1)), linestyle=linestyle, lw=2,
                         label=method_str + r', $\eta=%.3f$' % noise_level)

    plot_one(vi_results_all, vi_cmap, '--', 'Value iteration')
    plot_one(pi_results_all, pi_cmap, '-', 'Policy iteration')

    ax.legend()
    ax.set_xlabel(x_axis_type)
    ax.set_ylabel('Error')
    fig.tight_layout()
    return fig, ax


def relative_plot(xdata, ydata1, ydata2, method1=None, method2=None, ydata_type='', plot_type='scatter', pct_lwr=1, pct_upr=99, rat_lim=None,
                  fig=None, ax=None, save_plot=False):
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))
    num_trials = xdata.size
    scatter_alpha = min(10/num_trials**0.5, 1.0)

    ydata = ydata1 / ydata2

    # Axes limits
    xlim = [0, 1]
    if rat_lim is None:
        # Capture at least 98% of observations, snap outwards to next largest/smallest power of 10, and make symmetric
        yhardmin = 0.01
        yhardmax = 100.0
        ylogmin = min(np.log10(np.percentile(ydata, pct_lwr)), np.log10(yhardmin))
        ylogmax = max(np.log10(np.percentile(ydata, pct_upr)), np.log10(yhardmax))
        ylogabsmax = math.ceil(max(abs(ylogmin), abs(ylogmax)))
        ylim = (10**(-ylogabsmax*1.05), 10**(ylogabsmax*1.05))
    else:
        ylim = rat_lim

    # Colors
    if plot_type == 'scatter':
        color_good = 'C0'
        color_bad = 'C1'
        cdata = []
        for i in range(num_trials):
            if ydata[i] > 1.1:
                c = color_bad
            elif ydata[i] < 0.9:
                c = color_good
            else:
                c = 'k'
            cdata.append(c)
    elif plot_type == 'hexbin':
        cmap = 'Blues'
    else:
        raise ValueError

    if plot_type == 'scatter':
        ax.scatter(xdata, ydata, s=15, c=cdata, edgecolors='none', alpha=scatter_alpha)
        ax.axhline(y=1, color='k', linestyle='--', alpha=0.5)

        rect_upr = Rectangle((xlim[0], 1.0), xlim[1] - xlim[0], ylim[1] - 1.0, linewidth=1, edgecolor='none',
                             facecolor=color_bad, alpha=0.2)
        rect_lwr = Rectangle((xlim[0], ylim[0]), xlim[1] - xlim[0], 1.0 - ylim[0], linewidth=1, edgecolor='none',
                             facecolor=color_good, alpha=0.2)
        ax.add_patch(rect_upr)
        ax.add_patch(rect_lwr)
    elif plot_type == 'hexbin':
        ax.hexbin(xdata, ydata, yscale='log', bins='log', gridsize=gridsize, cmap=cmap,
                          extent=(xlim[0], xlim[1], np.log10(ylim[0]), np.log10(ylim[1])))

    ax.set_yscale('log')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if rat_lim is None:
        ax.set_yticks([10**(-ylogabsmax), 1.0, 10**(ylogabsmax)])
    ax.set_xlabel(r'$\eta$')
    # ax.set_title('k = %d'%(k))

    ax.set_ylabel('Ratio of ' + ydata_type)
    fig.tight_layout()

    if save_plot:
        delimiter = '_'
        ydata_type_s = ydata_type.replace(' ', delimiter)
        filename = delimiter.join(['many_systems', method1, method2, ydata_type_s, plot_type]) + '.pdf'
        plt.savefig('./figures/' + filename, dpi=600, format='pdf', bbox_inches='tight')

    return fig, ax
