"""Mechanism X — mutation rate modifier from fitness."""
import jax.numpy as jnp
from jax import jit

@jit
def mechanism_x(genomes, tags, fitness):
    fn = (fitness - jnp.min(fitness)) / (jnp.max(fitness) - jnp.min(fitness) + 1e-8)
    return 1.0 - fn * 0.5
