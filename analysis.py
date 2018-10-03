import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
from experiment_manager.explorer import make_index, get_immediate_subdirectories

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
    plt.errorbar(np.arange(0, t_max, test_every), train_mean, yerr=train_std, label='Train')
    plt.errorbar(np.arange(0, t_max, test_every), test_mean, yerr=test_std, label='Test')
    plt.legend()
    plt.savefig('{}/energies.png'.format(folder))
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
    for seed in range(n_seeds):
        eigenvalues_blob[:, :, seed] = np.load('{}/seed_{}/eigenvalues_acc.npy'.format(folder, seed))


    eigenvalues_blob /= np.mean(eigenvalues_blob)
    print(eigenvalues_blob[-3:, :, 0])

    # Make a "diffusion map" for the eigenvalues
    n_bins = 200
    v_min, v_max = 0, 3.5

    mixed_evs = np.clip(eigenvalues_blob.reshape(eigenvalues_blob.shape[0], -1), v_min, v_max)
    # print(np.sum(mixed_evs == 0), np.sum(mixed_evs == 1.5))
    bounds = [v_min, v_max]
    diff_map = np.zeros((eigenvalues_blob.shape[0], n_bins))
    for t in range(eigenvalues_blob.shape[0]):
        for ev in mixed_evs[t]:
            diff_map[t, int((n_bins-1) * (ev-bounds[0]) / (bounds[1] - bounds[0]))] += 1
            # print(int((n_bins-1) * (ev-bounds[0]) / (bounds[1] - bounds[0])))
        diff_map[t] /= np.sum(diff_map[t])

    # print(np.sum(diff_map[0]), np.sum(diff_map[-1]))
    # print(diff_map[0, :15])
    # print(diff_map[-1, :15])


    plt.figure()
    plt.imshow(diff_map.T, extent=[0, t_max, bounds[0], bounds[1]], aspect='auto')
    plt.colorbar()
    plt.savefig('{}/eigenvalues_diffmap.png'.format(folder))
    plt.close()



def parse_individual_subfolder(subfolder):
    # This function only acts as a container for other analysis routines (to keep things readable and modular)

    # make_average_energy(subfolder)
    make_eigenvalues_diffmap(subfolder)




def post_run_parsing():
    make_index()
    exp_dirs = get_immediate_subdirectories('out/raw')
    for exp_dir in exp_dirs:
        parse_individual_subfolder('out/raw/' + exp_dir)

if __name__ == '__main__':
    post_run_parsing()
