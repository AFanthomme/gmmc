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


    np.save('{}/E_train.npy'.format(folder), train_energy_blob)
    np.save('{}/E_train_avg.npy'.format(folder), train_mean)
    np.save('{}/E_train_std.npy'.format(folder), train_std)

    np.save('{}/E_test.npy'.format(folder), test_energy_blob)
    np.save('{}/E_test_avg.npy'.format(folder), test_mean)
    np.save('{}/E_test_std.npy'.format(folder), test_std)

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

    np.save('{}/mus.npy'.format(folder), mu_blob)
    np.save('{}/mu_avg.npy'.format(folder), mu_mean)
    np.save('{}/mu_std.npy'.format(folder), mu_std)

    plt.figure()
    plt.title(folder.split('/')[-1])
    plt.errorbar(np.arange(0, t_max, test_every), mu_mean, yerr=mu_std)
    plt.savefig('{}/mus.png'.format(folder))
    plt.close()


def summarize_likelihoods(folder):
    check_integrity(folder)

    with open('{}/params'.format(folder), 'r') as outfile:
        dict = json.load(outfile)
        n_seeds = dict['n_seeds']
        t_max = dict['t_max']
        test_every = dict['test_every']

    L_train_blob = np.zeros((n_seeds, t_max//test_every+1))
    L_test_blob = np.zeros((n_seeds, t_max//test_every+1))
    L_gen_blob = np.zeros((n_seeds, t_max//test_every+1))
    logZ_blob = np.zeros((n_seeds, t_max//test_every+1))
    Q2_blob = np.zeros((n_seeds, t_max//test_every+1))

    for seed in range(n_seeds):
        L_train_blob[seed] = np.load(folder + '/seed_{}/L_train_acc_ckpt.npy'.format(seed))
        L_test_blob[seed] = np.load(folder + '/seed_{}/L_test_acc_ckpt.npy'.format(seed))
        L_gen_blob[seed] = np.load(folder + '/seed_{}/L_gen_acc_ckpt.npy'.format(seed))
        logZ_blob[seed] = np.load(folder + '/seed_{}/logZ_acc_ckpt.npy'.format(seed))
        Q2_blob[seed] = np.load(folder + '/seed_{}/Q2_acc_ckpt.npy'.format(seed))

    np.save(folder + '/L_train', L_train_blob)
    np.save(folder + '/L_test', L_test_blob)
    np.save(folder + '/L_gen', L_gen_blob)
    np.save(folder + '/logZ', logZ_blob)
    np.save(folder + '/Q2', Q2_blob)


# def compute_likelihoods_a_posteriori(folder):
#     check_integrity(folder)
#
#     with open('{}/params'.format(folder), 'r') as outfile:
#         dict = json.load(outfile)
#         n_seeds = dict['n_seeds']
#         t_max = dict['t_max']
#         test_every = dict['test_every']
#         N = dict['n_neurons']
#         alpha = dict['alpha']
#         gamma = dict['gamma']
#
#     E_train_blob = np.zeros((n_seeds, t_max//test_every+1))
#     E_test_blob = np.zeros((n_seeds, t_max//test_every+1))
#     Q2_blob = np.zeros((n_seeds, t_max//test_every+1))
#
#     L_train_blob = np.zeros((n_seeds, t_max//test_every+1))
#     L_test_blob = np.zeros((n_seeds, t_max//test_every+1))
#     L_gen_blob = np.zeros((n_seeds, t_max//test_every+1))
#
#     for seed in range(n_seeds):
#         E_train_blob[seed] = np.load('{}/seed_{}/train_energy_acc.npy'.format(folder, seed))
#         E_test_blob[seed] = np.load('{}/seed_{}/test_energy_acc.npy'.format(folder, seed))
#         Q2_blob[seed] = (np.load('{}/seed_{}/eigenvalues_acc.npy'.format(folder, seed))**2).sum(axis=1)
#
#     # From this, recover L_train by using E = 1/N * (gamma/2 Q2 - alpha Ltrain)
#
#     L_train_blob = 1./alpha * (0.5*gamma*Q2_blob - alpha - N * E_train_blob)
#     L_test_blob = 1./alpha * (0.5*gamma*Q2_blob - alpha - N * E_test_blob)
#     L_gen_blob = L_train_blob - gamma / alpha * Q2_blob
#
#     np.save('{}/L_train.npy'.format(folder), L_train_blob)
#     np.save('{}/L_test.npy'.format(folder), L_test_blob)
#     np.save('{}/L_gen.npy'.format(folder), L_gen_blob)
#     np.save('{}/Q2.npy'.format(folder), Q2_blob)
#
#     # print(L_train_blob.mean(axis=0)[:-1].shape, L_train_blob.mean(axis=0)[:-1].shape, np.arange(0, t_max, test_every).shape)
#
#     plt.figure()
#     plt.title(folder.split('/')[-1])
#     plt.errorbar(np.arange(0, t_max, test_every), y=L_train_blob.mean(axis=0)[:-1], yerr=L_test_blob.std(axis=0)[:-1], label='L train')
#     plt.errorbar(np.arange(0, t_max, test_every), y=L_test_blob.mean(axis=0)[:-1], yerr=L_test_blob.std(axis=0)[:-1], label='L test')
#     plt.errorbar(np.arange(0, t_max, test_every), L_gen_blob.mean(axis=0)[:-1], yerr=L_gen_blob.std(axis=0)[:-1], label='L gen')
#     plt.legend()
#     plt.savefig('{}/likelihoods.png'.format(folder))
#     plt.close()





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

    np.save('{}/eigenvalues.npy'.format(folder), eigenvalues_blob.reshape((len(eigenvalues_blob), -1)))

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
    summarize_likelihoods(subfolder)
    make_average_mu(subfolder)
    make_eigenvalues_diffmap(subfolder)



def post_run_parsing(dir = 'out'):
    make_index(dir+ '/raw')

    exp_dirs = get_immediate_subdirectories(dir + '/raw')
    for exp_dir in exp_dirs:
        print('Treating folder {}'.format(exp_dir))
        parse_individual_subfolder(dir + '/raw/' + exp_dir)

if __name__ == '__main__':
    import sys
    try:
        dir = sys.argv[1]
        print(dir)
        post_run_parsing(dir)
    except IndexError:
        post_run_parsing()
