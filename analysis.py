import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
from experiment_manager.explorer import make_index, get_immediate_subdirectories

_DEBUG = False

def check_integrity(folder):
    try:
        with open('{}/exited_naturally'.format(folder), 'r') as f:
            pass
    except FileNotFoundError:
        print('Folder {} did not exit naturally, skip it'.format(folder))


def make_average_energy(folder):
    check_integrity(folder)

    with open('{}/params'.format(folder), 'r') as outfile:
        dict = json.load(outfile)
        n_seeds = dict['n_seeds']
        t_max = dict['t_max']
        test_every = dict['test_every']

    train_energy_blob = np.zeros((n_seeds, t_max//test_every+1))
    test_energy_blob = np.zeros((n_seeds, t_max // test_every + 1))

    for seed in range(n_seeds):
        train_energy_blob[seed] = np.load('{}/seed_{}/train_energy_acc.npy'.format(folder, seed))
        test_energy_blob[seed] = np.load('{}/seed_{}/test_energy_acc.npy'.format(folder, seed))

    train_mean = np.mean(train_energy_blob, axis=0)[1:]
    train_std = np.std(train_energy_blob, axis=0)[1:]

    test_mean = np.mean(test_energy_blob, axis=0)[1:]
    test_std = np.std(test_energy_blob, axis=0)[1:]


    plt.figure()
    plt.title(folder.split('/')[-1])
    plt.errorbar(np.arange(0, t_max, test_every), train_mean, yerr=train_std, label='Train')
    plt.errorbar(np.arange(0, t_max, test_every), test_mean, yerr=test_std, label='Test')
    plt.legend()
    plt.savefig('{}/energies.png'.format(folder))
    plt.close()


def make_average_mu(folder):
    check_integrity(folder)

    with open('{}/params'.format(folder), 'r') as outfile:
        dict = json.load(outfile)
        n_seeds = dict['n_seeds']
        t_max = dict['t_max']
        test_every = dict['test_every']

    mu_blob = np.zeros((n_seeds, t_max//test_every+1))


    for seed in range(n_seeds):
        mu_blob[seed] = np.load('{}/seed_{}/mu_acc.npy'.format(folder, seed))


    mu_mean = np.mean(mu_blob, axis=0)[1:]
    mu_std = np.std(mu_blob, axis=0)[1:]



    plt.figure()
    plt.title(folder.split('/')[-1])
    plt.errorbar(np.arange(0, t_max, test_every), mu_mean, yerr=mu_std)
    plt.savefig('{}/mus.png'.format(folder))
    plt.close()



def make_eigenvalues_diffmap(folder):
    check_integrity(folder)

    with open('{}/params'.format(folder), 'r') as outfile:
        dict = json.load(outfile)
        N = dict['n_neurons']
        n_seeds = dict['n_seeds']
        t_max = dict['t_max']
        test_every = dict['test_every']

    eigenvalues_blob = np.zeros((t_max // test_every + 1, N, n_seeds))
    mu_blob = np.zeros((t_max // test_every + 1, N, n_seeds))
    for seed in range(n_seeds):
        eigenvalues_blob[:, :, seed] = np.load('{}/seed_{}/eigenvalues_acc.npy'.format(folder, seed))
        mu_blob[:, :, seed] = np.repeat(np.load('{}/seed_{}/mu_acc.npy'.format(folder, seed)).reshape(-1, 1), N, axis=1)


    if _DEBUG:
        for t in range(1, 10):
            print(eigenvalues_blob[-t, :, 0].max().item(), eigenvalues_blob[-t, :, 0].min().item(), mu_blob[-t, 0, 0], mu_blob[-t, -1, 0], eigenvalues_blob[-t, :, 0].mean().item())

    eigenvalues_blob = mu_blob - eigenvalues_blob

    if _DEBUG:
        for t in range(1, 10):
            print(eigenvalues_blob[-t, :, 0].max().item(), eigenvalues_blob[-t, :, 0].min().item(), eigenvalues_blob[-t, :, 0].mean().item())


    # Make a "diffusion map" for the eigenvalues
    n_bins = 1000
    grouping = 10
    v_min, v_max = -5., 5.

    mixed_evs = np.clip(eigenvalues_blob.reshape(eigenvalues_blob.shape[0], -1), v_min, v_max)
    # print(np.min(mixed_evs), np.max(mixed_evs))
    # print(np.sum(mixed_evs == 0), np.sum(mixed_evs == 1.5))
    bounds = [v_min, v_max]
    diff_map = np.zeros((eigenvalues_blob.shape[0] // grouping +1, n_bins))

    for t in range(eigenvalues_blob.shape[0]):
        for ev in mixed_evs[t]:
            diff_map[t // grouping, int((n_bins-1) * (ev-bounds[0]) / (bounds[1] - bounds[0]))] += 1.

    for t in range(diff_map.shape[0]):
        diff_map[t] /= np.sum(diff_map[t])

    plt.figure()
    plt.imshow(diff_map[-100:].T, origin='lower', extent=[0, t_max, bounds[0], bounds[1]], aspect='auto')
    plt.title(folder.split('/')[-1])
    plt.savefig('{}/eigenvalues_diffmap.png'.format(folder))
    plt.close()



def parse_individual_subfolder(subfolder):
    # This function only acts as a container for other analysis routines (to keep things readable and modular)

    make_average_energy(subfolder)
    make_average_mu(subfolder)
    make_eigenvalues_diffmap(subfolder)



def post_run_parsing():
    make_index()
    exp_dirs = get_immediate_subdirectories('out/raw')
    for exp_dir in exp_dirs:
        print('Treating folder {}'.format(exp_dir))
        parse_individual_subfolder('out/raw/' + exp_dir)

if __name__ == '__main__':
    post_run_parsing()
