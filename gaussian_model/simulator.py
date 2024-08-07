import jax
import jax.numpy as jnp

class GaussianPrior():

    def __init__(self, mean_ms=0., scale_ms=1.):
        """
        A class to sample from the prior.

        Parameters
        ----------
        mean_ms : float, optional
            Mean of the prior. The default is 0.
        scale_ms : float, optional
            Scale of the prior. The default is 1.
        """
        self.mean_ms = mean_ms
        self.scale_ms = scale_ms

    def sample(self, rng_key, num_samples):
        """
        Samples num_samples from a Gaussian prior. A misspecification can
        be introduced by setting the mean and scale of the prior.

        Parameters
        ----------
        rng_key : jax.random.PRNGKey
            Random key for reproducibility.
        num_samples : int
            Number of samples to draw from the prior.
        mean_ms : float, optional
            Mean of the prior. The default is 0.
        scale_ms : float, optional
            Scale of the prior. The default is 1.

        Returns
        -------
        jnp.array
            Samples from the Gaussian prior.
        """
        mean = jnp.zeros((2,))+self.mean_ms
        cov = jnp.eye(2)*self.scale_ms
        return jax.random.multivariate_normal(rng_key, mean=mean, cov=cov, shape=(num_samples,))

class GaussianSimulator():
    def __init__(self, scale_ms=1., noise_ms=0.):
        """
        A class to simulate data from a Gaussian likelihood.

        Parameters
        ----------
        scale_ms : float, optional
            Scale of the covariance. The default is 1.
        noise_ms : float, optional
            Noise to be added to the simulation. The default is 0 (no noise).
        """
        self.scale_ms = scale_ms
        self.noise_ms = noise_ms

    def sample(self, params, rng_key, num_samples):
        """
        Simulate num_samples from a 2D Gaussian distribution given the mean as parameters. 
        A misspecification can be introduced via a scaling of the covariance or with additive noise.

        Parameters
        ----------
        params : jnp.array
            Mean of the Gaussian distribution.
        rng_key : jax.random.PRNGKey
            Random key for reproducibility.
        num_samples : int
            Number of samples to draw from the Gaussian distribution.

        Returns
        -------
        jnp.array
            Simulated samples from the Gaussian distribution given the parameters.
        """
        mean, cov = params, self.scale_ms * jnp.eye(2)
        z = jax.random.bernoulli(rng_key, self.noise_ms, shape=(num_samples,))
        z = jnp.hstack([z[:, None], z[:, None]])
        rng_key, _ = jax.random.split(rng_key)
        samples = jnp.where(
            z==1, jax.random.beta(rng_key, 2, 5, shape=(num_samples, 2)), jax.random.multivariate_normal(rng_key, mean=mean, cov=cov, shape=(num_samples,))
        )
        return samples

def calculate_analytic_posterior(prior, simulator, x):
    n_sim, D = x.shape

    # Set up variables
    sigma_0 = jnp.eye(D) * prior.scale_ms
    sigma_0_inv = jnp.linalg.inv(sigma_0)
    mu_0 = jnp.zeros((D, 1)) + prior.mean_ms
    sigma = simulator.scale_ms * jnp.eye(D)
    sigma_inv = jnp.linalg.inv(sigma)

    mu_posterior_covariance = jnp.stack([
        jnp.linalg.inv(sigma_0_inv+sigma_inv)]*n_sim)
    mu_posterior_mean = mu_posterior_covariance @ (sigma_0_inv @ mu_0 + sigma_inv @ x[..., jnp.newaxis])
    mu_posterior_mean = mu_posterior_mean.reshape(n_sim, D)

    return mu_posterior_mean, mu_posterior_covariance

class GenerativeModel():
    """
    GenerativeModel class that combines the prior and simulator to generate
    pairs of parameters and simulated samples.

    Parameters
    ----------
    prior : function
        Function to sample from the prior distribution.
    simulator : function
        Function to sample from the simulator given the parameters.
    """
    def __init__(self, prior, simulator):
        self.prior = prior
        self.simulator = simulator

    def generate(self, rng_key, num_samples):
        params = self.prior(rng_key, num_samples)
        rng_key, _ = jax.random.split(rng_key)
        samples = self.simulator(params, rng_key, num_samples)
        return (params, samples)