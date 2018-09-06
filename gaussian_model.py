import numpy as np

class CenteredGM:
    def __init__(self, dim, sigma=None):
        self.dim = dim
        if sigma is not None:
            self.sigma = sigma
        else:
            tmp = np.random.normal(size=[self.dim, self.dim])
            self.sigma = (tmp + tmp.T) / np.sqrt(2)
            self.sigma -= np.diag(np.diag(sigma))

        # Sanity checks before model build
        assert self.sigma.shape == [self.dim, self.dim]
        assert np.max(np.abs(self.sigma-self.sigma.T)) < 10**(-6)
        assert np.max(np.abs(np.diag(self.sigma))) < 10**(-6)

    def sample(self, n_samples):
        return np.random.multivariate_normal(mean=np.zeros(self.dim), cov=self.sigma, size=n_samples)

    def get_empirical_C(self, n_samples):
        obs = self.sample(n_samples)
        assert obs.shape == [n_samples, self.dim]
        return np.dot(obs.T, obs) / n_samples
