from gaussian_model import CenteredGM
from tqdm import tqdm
import numpy as np
import torch as tch
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
from scipy.optimize import brentq
from experiment_manager.explorer import get_siblings
import json
from MAP_theory import LikelihoodEstimator
import sys

N = 200

alphas = np.exp(np.linspace(*np.log([1.25, 40]), 30))
gammas = np.exp(np.linspace(*np.log([1e-2, 1e4]), 1000))

for coupling in [1e-3, 5e-3, 1e-2, 5e-2, 1e-1]:
    sigma = np.zeros((N, N))
    sigma[0, N - 1] = coupling
    sigma[N - 1, 0] = coupling
    for i in range(N - 1):
        sigma[i, i + 1] = coupling
        sigma[i + 1, i] = coupling

    model_to_fit = CenteredGM(N, precision=sigma, silent=False)
    sys.stdout.flush()
    explorer = LikelihoodEstimator(model_to_fit, n_batches=50, name='tridiag_' + 'j_{:.2e}_N_'.format(coupling) + '{}')
    explorer.grid_exploration(alphas, gammas)
