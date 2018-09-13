import numpy as np

def generate_pd_matrix(dim):
    sigma = np.random.normal(np.zeros([dim,dim], dtype=np.float32), 1./np.sqrt(dim))
    sigma = (sigma + sigma.T) / np.sqrt(2)
    sigma -= np.diag(np.diag(sigma))

    # Impose PD and mean of eigenvalues equal 1
    sigma += np.diag([dim for _ in range(dim)])
    sigma *= dim / np.sum(np.linalg.eig(sigma)[0])
    assert np.all(np.linalg.eig(sigma)[0] > 0.)
    assert np.abs(np.sum(np.linalg.eig(sigma)[0]) - dim) < 0.05

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
