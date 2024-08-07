import logging
from tqdm import tqdm

import jax
import jax.numpy as jnp

import optax

def update(params, opt_state, x, theta, loss_fn, optimizer):
    loss, grad = jax.value_and_grad(loss_fn)(params, x, theta)
    updates, opt_state = optimizer.update(grad, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss

class Trainer():
    """
    Class to train the model using a generative model.
    """

    def __init__(self, generative_model, model, loss_fn, learning_rate, checkpoint_path, max_to_keep=1):
        """
        Initializes the Trainer.

        Parameters
        ----------
        generative_model: generative_model.GenerativeModel
            Generative model.
        model: network.Network
            Network.
        loss_fn: function
            Loss function.
        learning_rate: float
            Learning rate.
        checkpoint_path: str
            Path to save the model.
        max_to_keep: int
            Maximum number of checkpoints to keep.
        """
        self.generative_model = generative_model
        self.model = model
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=learning_rate)
        )
        self.loss_fn = loss_fn
        self.checkpoint_path = checkpoint_path
        self.max_to_keep = max_to_keep

    def train(self, epochs, n_sims, batch_size, params, rng_key):
        """
        Trains the model.

        Parameters
        ----------
        epochs: int
            Number of epochs.
        rounds: int
            Number of rounds.
        sim_per_rounds: int
            Number of simulations per round.
        batch_size: int
            Batch size.
        """

        #Prepate logger
        logger = logging.getLogger()

        # Generate the data
        logger.info(f"Simulate {n_sims} data sets for training...")
        key, rng_key = jax.random.split(rng_key)
        theta, x = self.generative_model.generate_data(key, n_sims)

        train_loss = []
        validation_loss = []

        # Initialize the optimizer
        opt_state = self.optimizer.init(params)

        update_jit = jax.jit(update, static_argnums=(4, 5))

        # Train the model
        for epoch in tqdm(range(epochs), desc="Epochs"):
            logger.info(f"Epoch {epoch+1}/{epochs}")
            mean_loss = 0
            for i in range(0, n_sims, batch_size):
                mean_loss
                x_batch = x[i:i+batch_size]
                theta_batch = theta[i:i+batch_size]
                params, opt_state, loss = update(params, opt_state, x_batch, theta_batch, self.loss_fn, self.optimizer)
                mean_loss += loss
                