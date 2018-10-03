import tqdm
import numpy as np
import torch as tch
from gaussian_model import CenteredGM, generate_pd_matrix

import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar as minimize
from multiprocessing import Pool as ThreadPool
from analysis import post_run_parsing
from experiment_manager.launcher import explore_params, run_multi_threaded
from experiment_manager.explorer import make_index
import os

params = {
        # Simulation parameters
        'n_neurons': 50,
        'alpha': 3.,
        'gamma': 1,
        'beta_normalized': 1.,
        't_max': 400000,
        'n_seeds': 8,

        # method used to select reference C
        'which_C': 'bidiagonal',

        # Multi-threading and IO params
        'n_threads': 8,
        'silent': False,
        'test_every': 100,
        }

# Setting secondary parameters values
params['n_samples'] = int(params['alpha'] * params['n_neurons'])
params['beta'] = params['beta_normalized'] * params['n_neurons'] ** 2

params_to_vary = {'alpha' : [0.5, 2., 5., 7.]}
    #{'alpha' : [0.1, 1., 3., 10., 100.]}


def compute_energy(J, C, params):
    N = params['n_neurons']
    gamma = params['gamma']
    alpha = params['alpha']

    # J is the off-diagonal part
    spectrum, _ = tch.symeig(J, eigenvectors=True)

    # THE N² scaling will come from beta, so here everything should be order 1
    # Penalty part is straightforward
    l2_penalty = gamma * 0.25 * (spectrum**2).mean()

    # Likelihood part requires some calculations because of the spherical constraint:
    def surrogate(mu):
        return  mu - tch.log(mu-spectrum).mean().item()

    mu_opt = minimize(surrogate, bounds=(spectrum.max().item(), spectrum.max().item() + 15), method='bounded').x
    likelihood_energy = 0.5 * alpha * (tch.trace(tch.mm(J,C)) / N + surrogate(mu_opt))

    return l2_penalty + likelihood_energy, mu_opt - spectrum



def run_one_thread(out_dir, params, seed):
    # Make experiments different
    out_dir += 'seed_{}/'.format(seed)

    # If this fails, the whole exp should fail
    os.makedirs(out_dir)

    # Set seeds for reproducibility.
    # WARNING : GPU seeds are kinda weird https://discuss.pytorch.org/t/random-seed-initialization/7854/15
    np.random.seed(seed)
    tch.manual_seed(seed)
    if tch.cuda.is_available(): tch.cuda.manual_seed_all(seed)

    # Parameters unpacking
    N = params['n_neurons']
    alpha = params['alpha']
    t_max = params['t_max']
    beta = params['beta']
    test_every = params['test_every']

    if params['which_C'] == 'random':
        model_to_fit = CenteredGM(params['n_neurons'])
    elif params['which_C'] == 'bidiagonal':
        sigma = np.zeros((N, N))
        for i in range(N - 1):
            sigma[i, i + 1] = 1
            sigma[i + 1, i] = 1
        model_to_fit = CenteredGM(N, sigma=sigma)

    # Generate two empirical C to check overfitting
    C_train = tch.from_numpy(model_to_fit.get_empirical_C(n_samples= alpha * N).astype(np.float32))
    C_test = tch.from_numpy(model_to_fit.get_empirical_C(n_samples= alpha * N).astype(np.float32))


    # Initialization for J (sym, gaussian, zero diag)
    J = tch.normal(mean=tch.zeros([N,N]), std=1./np.sqrt(N) * tch.ones([N,N]))
    J = (J + J.t()) / np.sqrt(2)
    J -= tch.diag(tch.diag(J))


    # Initialize the accumulators
    train_energy_acc = np.zeros(t_max//test_every+1)
    test_energy_acc = np.zeros(t_max // test_every + 1)
    eigenvalues_acc = np.zeros((t_max//test_every+1, N))

    # The eigenvalues we store are those of (mu*In - J) to get the "only positive evs"
    energy, eigenvalues = compute_energy(J, C_train, params)
    train_energy_acc[0] = energy
    eigenvalues_acc[0] = eigenvalues

    current_energy = energy

    # MC loop
    for t in tqdm.tqdm(range(1, t_max)):
        # Propose a change
        i, j = np.random.randint(N, size=2)
        epsilon = np.random.normal(scale=1./np.sqrt(N))

        J_prop = J.clone()
        J_prop[i, j] += epsilon
        J_prop[j, i] += epsilon

        F_prop, spectrum = compute_energy(J_prop, C_train, params)

        if np.isnan(F_prop):
            print("Invalid move")
            continue

        delta_F = F_prop - current_energy
        # To check that beta delta_f is reasonable
        # print(-beta*delta_F)

        if delta_F < 0:
            current_energy = F_prop
            J = J_prop.clone()
        elif np.random.rand() < np.exp(-beta*delta_F):
            current_energy = F_prop
            J = J_prop.clone()
        else:
            pass

        # Introduce checkpoints !
        if t % test_every == (test_every - 1):
            idx = t // test_every + 1 # idx 0 is before starting
            eigenvalues_acc[idx] = spectrum

            train_energy_acc[idx] = current_energy
            plt.figure()
            plt.plot(train_energy_acc[:t])
            plt.savefig(out_dir+'train_energy_real_time.png')
            plt.close()

            test_energy_acc[idx] = compute_energy(J, C_test, params)[0]
            plt.figure()
            plt.plot(train_energy_acc[:t])
            plt.savefig(out_dir+'test_energy_real_time.png')
            plt.close()

            if idx % 50 == 49:
                np.save(out_dir+'train_energy_acc_ckpt', train_energy_acc)
                np.save(out_dir + 'test_energy_acc_ckpt', test_energy_acc)
                np.save(out_dir+'eigenvalues_acc_ckpt', eigenvalues_acc)

    np.save(out_dir+'train_energy_acc', train_energy_acc)
    np.save(out_dir+'test_energy_acc', test_energy_acc)
    np.save(out_dir+'eigenvalues_acc', eigenvalues_acc)

    return


if __name__ == '__main__':
    explore_params(run_one_thread, params, params_to_vary)
    post_run_parsing()



