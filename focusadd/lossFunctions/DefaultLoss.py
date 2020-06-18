from .LossFunction import LossFunction
import jax.numpy as np
import math
from jax import jit
from functools import partial
from jax.config import config
config.update("jax_enable_x64", True)

PI = math.pi


def default_loss(surface_data, params_to_data, w_args, params):
    """ 
	Computes the default loss: int (B dot n)^2 dA + weight_length * len(coils) 

	Input: params, a tuple of the fourier series for the coils and a fourier series for the rotation.

	Output: A scalar, which is the loss_val computed by the function. JAX will eventually differentiate
	this in an optimizer.
	"""
    w_B, w_L = w_args
    r_surf_central, nn, sg = surface_data
    I, dl, _, r_midpoint, total_length = params_to_data(params)

    B_loss_val = LossFunction.quadratic_flux(
            r_surf_central,
            I,
            dl,
            r_midpoint,
            nn,
            sg,
        )

    return w_B * B_loss_val + w_L * total_length
