import os
import warnings

import jax

from erbs.utils import setup_ase

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
jax.config.update("jax_enable_x64", True)

setup_ase()

warnings.filterwarnings(action="ignore", category=FutureWarning, module=r"jax.*scatter")
