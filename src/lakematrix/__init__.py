"""lakematrix: the LAKE thermodynamic core with the lake as an array dimension."""

import jax

# The model is float64 throughout (plan section 0, decision 1).
jax.config.update("jax_enable_x64", True)

__version__ = "0.1.0"
