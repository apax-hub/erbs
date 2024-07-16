import os
import warnings

import jax

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
warnings.filterwarnings(action="ignore", category=FutureWarning, module=r"jax.*scatter")
jax.config.update("jax_enable_x64", True)
