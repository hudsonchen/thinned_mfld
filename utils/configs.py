# ----------------------- Config -----------------------
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax import jit, vmap, grad, random, lax
from jaxtyping import Array 

@dataclass
class CFG:
    N: int = 2048
    steps: int = 5000
    step_size: float = 1e-3
    sigma: float = 1.0
    zeta : float = 1e-2
    seed: int = 0
    kernel: str = "sobolev"  # "sobolev" or "gaussian"
    g : int = 0  # parameter for KT thinning
    bandwidth: float = 1.0  # for Gaussian kernel
    # Return full trajectory if True (memory heavy): (steps+1, N, d)
    return_path: bool = False

# ----------------------- Problem spec -----------------------

from dataclasses import dataclass

@dataclass
class MFC_Config:
    # sizes
    N: int = 256       # particles
    d: int = 2         # state dimension
    a_dim: int = 2     # action dimension

    # time
    T: float = 1.0
    dt: float = 0.01
    sigma: float = 0.3

    # kernel
    bandwidth: float = 1.0

    # SGD
    K_sgd: int = 200
    lr: float = 1e-3
    seed: int = 0
    log_every: int = 20
