import jax
import jax.numpy as jnp
import flax.linen as nn

class MLPCompressor(nn.Module):
    """
    MLPCompressor is a simple MLP that compresses the input data.
    """
    hidden_size: list
    output_size: int

    @nn.compact
    def __call__(self, x):
        for size in self.hidden_size:
            x = nn.Dense(size)(x)
            x = nn.relu(x)
        x = nn.Dense(self.output_size)(x)
        return x
    
class CNN2DCompressor(nn.Module):
    output_dim: int
    activation: callable

    @nn.compact
    def __call__(self, inputs):
        net_x = nn.Conv(32, 3, 2)(inputs)
        net_x = self.activation(net_x)
        net_x = nn.Conv(64, 3, 2)(net_x)
        net_x = self.activation(net_x)
        net_x = nn.Conv(128, 3, 2)(net_x)
        net_x = self.activation(net_x)
        net_x = nn.avg_pool(net_x, (16, 16), (8, 8), padding='SAME')
        net_x = net_x.reshape((net_x.shape[0], -1)) #flatten
        net_x = nn.Dense(64)(net_x)
        net_x = self.activation(net_x)
        net_x = nn.Dense(self.output_dim)(net_x)

        return net_x.squeeze()
                             
    
class MAF_MLPCompressor(nn.Module):
    mlp_compressor: nn.Module
    nf: nn.Module
    mlp_hparams: dict
    nf_hparams: dict

    def setup(self):
        self.mlp_compressor_nn = self.mlp_compressor(**self.mlp_hparams)
        self.nf_nn = self.nf(**self.nf_hparams)

    def __call__(self, x, theta):
        """
        Computes the log-probability for Neural Posterior Estimation.

        Parameters
        ----------
        x: jnp.array
            Simulated data.
        theta: jnp.array
            Input parameters.
        
        Returns
        -------
        jnp.array
            Log-probability.
        """
        z = self.mlp_compressor_nn(x)
        log_prob = self.nf_nn.log_prob(theta, z)
        return log_prob
    
    def log_prob_nle(self, x, theta):
        """
        Computes the log-probability for Neural Likelihood Estimation.

        Parameters
        ----------
        x: jnp.array
            Simulated data.
        theta: jnp.array
            Input parameters.
        
        Returns
        -------
        jnp.array
            Log-probability.
        """
        z = self.mlp_compressor_nn(x)
        log_prob = self.nf_nn.log_prob(z, theta)
        return log_prob
    
    def compress(self, x):
        """
        Compresses the input data.

        Parameters
        ----------
        x: jnp.array
            Input data.
        
        Returns
        -------
        jnp.array
            Compressed data.
        """
        return self.mlp_compressor_nn(x)
    
    def log_prob_from_compressed(self, theta, z):
        """
        Computes the log-probability given the compressed data.

        Parameters
        ----------
        theta: jnp.array
            Input parameters.
        z: jnp.array
            Compressed data.
        
        Returns
        -------
        jnp.array
            Log-probability.
        """
        return self.nf_nn.log_prob(theta, z)
    
    def log_prob_from_compressed_nle(self, z, theta):
        """
        Computes the log-probability given the compressed data.

        Parameters
        ----------
        z: jnp.array
            Compressed data.
        theta: jnp.array
            Input parameters.
        
        Returns
        -------
        jnp.array
            Log-probability.
        """
        return self.nf_nn.log_prob(z, theta)
    
    def sample(self, rng_key, x, n_samples):
        """
        Samples from the model.

        Parameters
        ----------
        rng_key: jax.random.PRNGKey
            Random key for reproducibility.
        x: jnp.array
            Observed data.
        n_samples: int
            Number of samples to draw.
        
        Returns
        -------
        jnp.array
            Samples from the distribution represented by the Normalizing Flow.
        """
        z = self.mlp_compressor_nn(x)
        return self.nf_nn.sample(z, n_samples, rng_key)

#The code is redundent it could be embed in one unique class   
class MAF_CNNCompressor(nn.Module):
    cnn_compressor: nn.Module
    nf: nn.Module
    cnn_hparams: dict
    nf_hparams: dict

    def setup(self):
        self.cnn_compressor_nn = self.cnn_compressor(**self.cnn_hparams)
        self.nf_nn = self.nf(**self.nf_hparams)

    def __call__(self, x, theta):
        """
        Computes the log-probability for Neural Posterior Estimation.

        Parameters
        ----------
        x: jnp.array
            Simulated data.
        theta: jnp.array
            Input parameters.
        
        Returns
        -------
        jnp.array
            Log-probability.
        """
        z = self.cnn_compressor_nn(x)
        log_prob = self.nf_nn.log_prob(theta, z)
        return log_prob
    
    def log_prob_nle(self, x, theta):
        """
        Computes the log-probability for Neural Likelihood Estimation.

        Parameters
        ----------
        x: jnp.array
            Simulated data.
        theta: jnp.array
            Input parameters.
        
        Returns
        -------
        jnp.array
            Log-probability.
        """
        z = self.cnn_compressor_nn(x)
        log_prob = self.nf_nn.log_prob(z, theta)
        return log_prob
    
    def compress(self, x):
        """
        Compresses the input data.

        Parameters
        ----------
        x: jnp.array
            Input data.
        
        Returns
        -------
        jnp.array
            Compressed data.
        """
        return self.cnn_compressor_nn(x)
    
    def log_prob_from_compressed(self, theta, z):
        """
        Computes the log-probability given the compressed data.

        Parameters
        ----------
        theta: jnp.array
            Input parameters.
        z: jnp.array
            Compressed data.
        
        Returns
        -------
        jnp.array
            Log-probability.
        """
        return self.nf_nn.log_prob(theta, z)
    
    def log_prob_from_compressed_nle(self, z, theta):
        """
        Computes the log-probability given the compressed data.

        Parameters
        ----------
        z: jnp.array
            Compressed data.
        theta: jnp.array
            Input parameters.
        
        Returns
        -------
        jnp.array
            Log-probability.
        """
        return self.nf_nn.log_prob(z, theta)
    
    def sample(self, rng_key, x, n_samples):
        """
        Samples from the model.

        Parameters
        ----------
        rng_key: jax.random.PRNGKey
            Random key for reproducibility.
        x: jnp.array
            Observed data.
        n_samples: int
            Number of samples to draw.
        
        Returns
        -------
        jnp.array
            Samples from the distribution represented by the Normalizing Flow.
        """
        z = self.cnn_compressor_nn(x)
        return self.nf_nn.sample(z, n_samples, rng_key)