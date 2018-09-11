import numpy as np

def generate_pd_matrix(dim):
    # Use cholevsky factorization (enough for only generation)
    # ref : http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.31.494&rep=rep1&type=pdf
    count = 0
    while True:
        theta = np.random.normal(size=[dim, dim])
        count += 1
        if np.linalg.matrix_rank(theta, tol=10**(-8)) == dim:
            print("generating pd matrix, retry #{}".format(count))
            break
    theta = np.triu(theta)
    sigma = np.dot(theta.T, theta)

    # Legacy
    # tmp = np.random.normal(size=[dim, dim])
    # sigma = (tmp + tmp.T) / np.sqrt(2)
    # sigma -= np.diag(np.diag(sigma))
    # sigma -= np.diag([np.min(np.min(np.linalg.eig(sigma)[0]), 0) -0.1 for _ in range(dim)])

    # To avoid issues with very small negative ev
    sigma += np.diag([0.001 for _ in range(dim)])
    assert np.all(np.linalg.eig(sigma)[0] > 0.)
    return sigma

class CenteredGM:
    def __init__(self, dim, sigma=None):
        self.dim = dim
        if sigma is not None:
            self.sigma = sigma
        else:
            self.sigma = generate_pd_matrix(self.dim)

        # Sanity checks before model build
        # print(np.linalg.eig(self.sigma)[0])
        assert np.all(np.linalg.eig(self.sigma)[0] > 0.)
        assert self.sigma.shape == (self.dim, self.dim)
        assert np.max(np.abs(self.sigma-self.sigma.T)) < 10**(-6)
        # assert np.max(np.abs(np.diag(self.sigma))) < 10**(-6)

    def sample(self, n_samples):
        return np.random.multivariate_normal(mean=np.zeros(self.dim), cov=self.sigma, size=n_samples)

    def get_empirical_C(self, n_samples):
        obs = self.sample(n_samples)
        assert obs.shape == (n_samples, self.dim)
        assert np.dot(obs.T, obs).shape == (self.dim, self.dim)
        return np.dot(obs.T, obs) / n_samples
