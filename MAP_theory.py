from gaussian_model import CenteredGM
from tqdm import tqdm
import numpy as np
import torch as tch
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from experiment_manager.explorer import get_siblings, get_immediate_subdirectories
from scipy.optimize import curve_fit
import json
import os

class LikelihoodEstimator:
    def __init__(self, gaussian_model, alpha=3., n_batches=1, name='tridiag{}'):
        self.ref_model = gaussian_model
        self.N = self.ref_model.dim
        self.name = name.format(self.N)
        self.alpha = alpha

        self.n_batches = n_batches
        self.C_true = tch.from_numpy(self.ref_model.covariance)

        # precision is actually the full precision matrix (mu - J)^true, so split it here
        self.J_true = -tch.from_numpy(self.ref_model.precision - np.diag(np.diag(self.ref_model.precision)))
        self.mu_true = np.mean(np.diag(self.ref_model.precision))

        # Twice the mu of gamma=0
        self.mu_lim = 2 * alpha/(alpha-1.)*self.ref_model.precision[0, 0]

        # Commodity to have it here
        self.labels = ['L_train', 'L_test', 'L_gen', 'Q2', 'logZ', 'mu', 'errors_on', 'errors_off', 'mu_dot', 'gamma_cros_res', 'L_test_dot']
        self._make_output_dir()

    def _make_output_dir(self):
        try:
            with open('saves_for_MAP/{}/test_out_exists.txt'.format(self.name)) as f:
                pass
        except FileNotFoundError:
            os.makedirs('saves_for_MAP/{}'.format(self.name))
            with open('saves_for_MAP/{}/test_out_exists.txt'.format(self.name), mode='w+') as f:
                f.write('plop')

    def j_map(self, eigs, gamma, mu):
        a = self.alpha
        return (a*eigs + gamma*mu - tch.sqrt((a*eigs - gamma*mu)**2 + 4 * a * gamma)) / (2. * gamma)

    def solve_mu(self, gamma, eigs):
        def surrogate(mu):
            return 1. - tch.mean(1. / (mu - self.j_map(eigs, gamma, mu))).item()

        try:
            root = brentq(surrogate, 0.95, self.mu_lim + 1.)
        except ValueError:
            print('Solving in the prescribed range failed')
            return None

        return root

    def do_one_round(self, gamma_range):
        # Draw the covariance matrix1
        C_emp = tch.from_numpy(self.ref_model.get_empirical_C(self.alpha * self.N))
        # C is the vector of eigenvalues of C_emp
        C, BasisChange = tch.symeig(C_emp, eigenvectors=True)

        # This part for educational purposes

        # B is orthogonal
        # print(B.mm(B.transpose(0, 1)))

        # B is NOT symmetrical
        # print(tch.max(B - B.transpose(0, 1)))

        # B is normed
        # print(B.shape, tch.sum(B**2, dim=0))

        # When alpha is huge, C_emp-C_true is small in infty norm
        # print(tch.max(self.C_true - C_emp))

        # Therefore, in that case, diagonalizing C_emp should almost diagonalize C_true
        # print(B.transpose(0, 1).mm(self.C_true).mm(B))

        # Prepare all accumulators...
        L_train, L_test, Q2, L_gen, logZ, mus = [], [], [], [], [], []
        errors_on, errors_off = [], []
        mu_dot, gamma_cross_res, L_test_dot = [], [], []

        # Pre-compute some stuff to win time (and readanility)
        alpha_C = self.alpha * C
        alpha = self.alpha

        for gamma in gamma_range:
            mu = self.solve_mu(gamma, C)
            if mu is None:
                return None

            mus.append(mu)

            j_star = self.j_map(C, gamma, mu)
            mu_minus_j = mu - j_star

            # The easy ones :
            L_train.append(.5 * tch.mean(C * j_star))
            Q2.append(.5 * tch.mean(j_star ** 2))
            logZ.append(.5 * (mu - tch.mean(tch.log(mu_minus_j))))
            L_gen.append(L_train[-1] - gamma / self.alpha * Q2[-1])

            # L_test is a bother : we know J_map only in the C_emp diagonalizing basis
            # Therefore, need to rotate C_true in the inference basis (keep only diag, rest is not useful)
            C_hat = tch.diag(BasisChange.transpose(0, 1).mm(self.C_true).mm(BasisChange))
            L_test.append(0.5 * tch.mean(C_hat * j_star))

            # Now for the model errors (computed in the true model basis:
            J_star_in_true_basis = BasisChange.mm(tch.diag(j_star)).mm(BasisChange.transpose(0, 1))
            J_star_on_support = tch.where(self.J_true != 0, J_star_in_true_basis, tch.zeros(self.N, self.N).double())
            J_star_off_support = tch.where(self.J_true != 0, tch.zeros(self.N, self.N).double(), J_star_in_true_basis)

            # To compute meaningful means
            support_size = tch.sum(tch.where(self.J_true != 0, tch.ones(self.N, self.N).double(), tch.zeros(self.N, self.N).double())).item()
            # print(support_size)

            errors_on.append(tch.sum((J_star_on_support-self.J_true)**2)/support_size)
            errors_off.append(tch.sum((J_star_off_support)**2)/(self.N ** 2 - support_size))

            # Now, the gamma_cross "residual" -> = 0 at crossing
            gamma_cross_res.append(gamma - self.alpha * tch.mean(j_star * (C-C_hat)) / (2*Q2[-1]))

            # Now, the gory stuff: find mu_dot and L_test_dot
            D = np.sqrt((alpha_C-gamma*mu)**2+4*gamma*alpha)

            V = mu - (2*alpha + gamma * (mu**2) - alpha_C * mu)/ D
            V /= (2.*gamma)

            W = 1. - (gamma*mu - alpha_C) / D
            W /= 2.

            _n = tch.mean((V - j_star / gamma) / (mu_minus_j**2))
            _d = tch.mean((1.-W) / (mu_minus_j**2))

            mu_dot.append(_n/_d)

            j_dot = mu_dot[-1] * W - j_star / gamma + V
            j_dot_term = 0.5 * tch.mean(j_dot*C_hat)
            logZ_dot = .5 * (mu_dot[-1] - tch.mean((mu_dot[-1]-j_dot)/(mu-j_star)))
            L_test_dot.append(j_dot_term - logZ_dot)


        # Wrap results in a dict for cleaner code
        out = {}
        quantities = [L_train, L_test, L_gen, Q2, logZ, mus, errors_on,
                        errors_off, mu_dot, gamma_cross_res, L_test_dot]

        for key, value in zip(self.labels, quantities):
            out[key] = value

        return out

    def gamma_exploration(self, gamma_range, verbose=False):
        idx = 0
        while idx < self.n_batches:
            one_round_result = self.do_one_round(gamma_range)
            if one_round_result is None:
                continue
            if idx == 0:
                accumulator = {key: tch.zeros(len(gamma_range), self.n_batches) for key in one_round_result}

            for key, value in one_round_result.items():
                accumulator[key][:, idx] = tch.tensor(value)

            idx += 1

            if idx % 10 == 0 and verbose:
                print('Finished pass {} over {} \r'.format(idx, self.n_batches))

        return accumulator

    def grid_exploration(self, alpha_range, gamma_range, plot=True, verbose=False):
        means_acc = {}
        std_acc = {}

        additional_quantities = ['crossing', 'half_crossing', 'lkl_ratio', 'crossings_pred', 'gamma_opt_pred', 'gamma_opt']

        for key in self.labels + additional_quantities:
            means_acc[key] = []
            std_acc[key] = []

        # crossings_acc = {'mean': [], 'std': []}

        for alpha in alpha_range:
            self.alpha = alpha
            acc = self.gamma_exploration(gamma_range, verbose=verbose)
            crossings = []
            half_crossings = []
            lkl_ratio = []
            crossings_pred = []
            gamma_opt_pred = []
            gamma_opt = []

            for key, value in acc.items():
                value = value.cpu().numpy()
                means_acc[key].append(value.mean(axis=-1))
                std_acc[key].append(value.std(axis=-1))

            for i in range(self.n_batches):
                crossings.append(gamma_range[np.argmin(np.abs((acc['L_gen'] - acc['L_test'])/(acc['L_train'] - acc['L_test']))[:, i])])
                half_crossings.append(gamma_range[np.argmin(np.abs((acc['L_gen'] - acc['L_test'])/(acc['L_train'] - acc['L_test'])-0.5)[:, i])])
                gamma_opt.append(gamma_range[np.argmax((acc['L_test']-acc['logZ'])[:, i])])
                # Several points where it goes to 0 (local minima), could probably be improves
                gamma_opt_pred.append(gamma_range[np.argmin(3 * (gamma_range > 3. ) + 3 * (gamma_range < 1e-2) + np.abs(gamma_range * acc['L_test_dot'][:, i]))])
                crossings_pred.append(gamma_range[np.argmin(np.abs(acc['gamma_cros_res'])[:, i])])
                # print(gamma_opt[-1], gamma_opt_pred[-1])

            means_acc['lkl_ratio'].append(tch.mean((acc['L_gen'] - acc['L_test'])/(acc['L_train'] - acc['L_test']), dim=-1).cpu().numpy())
            std_acc['lkl_ratio'].append(tch.std((acc['L_gen'] - acc['L_test'])/(acc['L_train'] - acc['L_test']), dim=-1).cpu().numpy())

            means_acc['crossing'].append(np.mean(crossings))
            std_acc['crossing'].append(np.std(crossings))

            means_acc['half_crossing'].append(np.mean(half_crossings))
            std_acc['half_crossing'].append(np.std(half_crossings))

            means_acc['gamma_opt'].append(np.mean(gamma_opt))
            std_acc['gamma_opt'].append(np.std(gamma_opt))

            means_acc['gamma_opt_pred'].append(np.mean(gamma_opt_pred))
            std_acc['gamma_opt_pred'].append(np.std(gamma_opt_pred))

            means_acc['crossings_pred'].append(np.mean(crossings_pred))
            std_acc['crossings_pred'].append(np.std(crossings_pred))

            if plot:
                fig, axes = plt.subplots(4, 2, figsize=(12, 13))

                axes[0,0].set_title(r'$\mu$')
                axes[0,0].errorbar(gamma_range, means_acc['mu'][-1], yerr=std_acc['mu'][-1], c='b')
                axes[0,0].set_xscale('log')

                # The multiplication by gamma is here to switch from d gamma to d loggamma
                axes[0,1].set_title(r'Derivatives wrt $log\gamma$')
                axes[0,1].errorbar(gamma_range, gamma_range * means_acc['mu_dot'][-1], yerr=gamma_range * std_acc['mu_dot'][-1], c='g', label='mu')
                axes[0,1].errorbar(gamma_range, gamma_range * means_acc['L_test_dot'][-1], yerr=gamma_range * std_acc['L_test_dot'][-1], c='r', label='L_test')
                axes[0,1].set_xscale('log')
                axes[0,1].legend()

                th_acc_substracted_logZ = {}
                for key in ['L_train', 'L_test', 'L_gen']:
                    th_acc_substracted_logZ[key] = means_acc[key][-1] - means_acc['logZ'][-1]

                axes[1,0].set_title('Likelihoods (without logZ)')
                axes[1,0].errorbar(gamma_range, means_acc['L_train'][-1], yerr=std_acc['L_train'][-1], c='b', label='train')
                axes[1,0].errorbar(gamma_range, means_acc['L_test'][-1], yerr=std_acc['L_test'][-1], c='r', label='test')
                axes[1,0].errorbar(gamma_range, means_acc['L_gen'][-1], yerr=std_acc['L_gen'][-1], c='g', label='gen')
                axes[1,0].set_xscale('log')

                axes[1,1].set_title('Likelihoods (with logZ)')
                axes[1,1].errorbar(gamma_range, th_acc_substracted_logZ['L_train'], yerr=std_acc['L_train'][-1], c='b', label='train')
                axes[1,1].errorbar(gamma_range, th_acc_substracted_logZ['L_test'], yerr=std_acc['L_test'][-1], c='r', label='test')
                axes[1,1].errorbar(gamma_range, th_acc_substracted_logZ['L_gen'], yerr=std_acc['L_gen'][-1], c='g', label='gen')
                axes[1,1].set_xscale('log')

                axes[2,0].set_title('Likelihoods_ratio')
                axes[2,0].errorbar(gamma_range, means_acc['lkl_ratio'][-1], yerr=std_acc['lkl_ratio'][-1], c='b')
                axes[2,0].set_xscale('log')

                axes[2,1].set_title('Q2')
                axes[2,1].errorbar(gamma_range, means_acc['Q2'][-1], yerr=std_acc['Q2'][-1], c='b')
                axes[2,1].set_xscale('log')

                axes[3,0].set_title('On-support errors')
                axes[3,0].errorbar(gamma_range, means_acc['errors_on'][-1], yerr=std_acc['errors_on'][-1], c='b')
                axes[3,0].set_xscale('log')

                axes[3,1].set_title('Off-support errors')
                axes[3,1].errorbar(gamma_range, means_acc['errors_off'][-1], yerr=std_acc['errors_off'][-1], c='r')
                axes[3,1].set_xscale('log')


                fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                fig.suptitle(r'Summary for $\alpha$ = {}'.format(alpha)+'\n ')
                fig.savefig('saves_for_MAP/{}/full_figs_{:.2f}.pdf'.format(self.name, self.alpha))


        plt.figure()
        plt.title(r'Notable gammas as a function of $\alpha$')
        plt.xlabel(r'$\alpha$')
        plt.errorbar(alpha_range, means_acc['crossing'], yerr=std_acc['crossing'], label=r'$\gamma^{cross}$ From curves')
        plt.errorbar(alpha_range, means_acc['crossings_pred'], yerr=std_acc['crossings_pred'], label=r'$\gamma^{cross}$ From residual')
        plt.errorbar(alpha_range, means_acc['gamma_opt'], yerr=std_acc['gamma_opt'], label=r'$\gamma^{opt}$ From curves')
        plt.errorbar(alpha_range, means_acc['gamma_opt_pred'], yerr=std_acc['gamma_opt_pred'], label=r'$\gamma^{opt}$ From residual')
        plt.legend()
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig('saves_for_MAP/{}/crossings.pdf'.format(self.name))
        plt.close()

        plt.figure()
        plt.title('Test/Gen half_crossing as a function of alpha')
        plt.xlabel(r'$\alpha$')
        plt.ylabel(r'$\gamma^{half crossing}$')
        plt.errorbar(alpha_range, means_acc['half_crossing'], yerr=std_acc['half_crossing'])
        plt.savefig('saves_for_MAP/{}/half_crossings.pdf'.format(self.name))
        plt.close()

        np.savez('saves_for_MAP/{}/likelihoods_means.npz'.format(self.name), **means_acc)
        np.savez('saves_for_MAP/{}/likelihoods_std.npz'.format(self.name), **std_acc)
        np.savetxt('saves_for_MAP/{}/alphas.txt'.format(self.name), alpha_range)
