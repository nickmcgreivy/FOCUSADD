from .LossFunction import LossFunction
import jax.numpy as np
import math
from jax import jit
from functools import partial

PI = math.pi


def default_loss(surface_data, params_to_data, weight_length, params):
    """ 
	Computes the default loss: int (B dot n)^2 dA + weight_length * len(coils) 

	Input: params, a tuple of the fourier series for the coils and a fourier series for the rotation.

	Output: A scalar, which is the loss_val computed by the function. JAX will eventually differentiate
	this in an optimizer.
	"""

    r_surf_central, nn, sg = surface_data


    # NEED TO SET_PARAMS
    I, dl, _, r_middle, total_length = params_to_data(params)
    B_loss_val = np.sum(
        LossFunction.bnsquared(
            r_surf_central,
            I,
            dl,
            r_middle,
            nn,
            sg,
        )
    )

    return B_loss_val + weight_length * total_length
