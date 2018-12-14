import numpy as np
import matplotlib.pyplot as plt
from experiment_manager.explorer import get_siblings, get_immediate_subdirectories
from scipy.optimize import curve_fit




def log_crossing_analysis(dir):
    def linear(x, a, b):
        return a*x+b

    means = np.load('saves_for_theory/{}/likelihoods_means.npz'.format(dir))['crossing'][:10]
    std = np.load('saves_for_theory/{}/likelihoods_std.npz'.format(dir))['crossing'][:10]
    alphas = np.loadtxt('saves_for_theory/{}/alphas.txt'.format(dir))[:10]

    popt = curve_fit(linear, np.log(alphas), np.log(means))[0]

    plt.figure()
    plt.title(dir+'\n Exponent {:.2e}, prefactor {:.2e}'.format(popt[0], np.exp(popt[1])))
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$\gamma$ at test/gen crossing')
    plt.errorbar(alphas, means, yerr=std, label='theory')
    plt.plot(alphas, np.exp(linear(np.log(alphas), *popt)), label='fit')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('saves_for_theory/{}/log_crossings.pdf'.format(dir))
    plt.show()

def post_run_analysis():
    # Hypothesis : when J increases, the exponent gets more negative and the prefactor gets smaller
    subdirs = get_immediate_subdirectories('saves_for_theory')
    subdirs.sort()

    for dir in subdirs:
        print(dir)
        log_crossing_analysis(dir)


if __name__ == '__main__':
    post_run_analysis()
