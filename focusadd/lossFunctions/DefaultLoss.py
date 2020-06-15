from .LossFunction import LossFunction
import jax.numpy as np
import math
from jax import jit

PI = math.pi



# @partial(jit, static_argnums=(0,))
def default_loss(surface_data, coil_set, weight_length, params):
    """ 
	Computes the default loss: int (B dot n)^2 dA + weight_length * len(coils) 

	Input: params, a tuple of the fourier series for the coils and a fourier series for the rotation.

	Output: A scalar, which is the loss_val computed by the function. JAX will eventually differentiate
	this in an optimizer.
	"""

    r_central, nn, sg = surface_data

    # NEED TO SET_PARAMS
    coil_set.set_params(params)
    B_loss_val = np.sum(
        LossFunction.bnsquared(
            r_central,
            coil_set.get_I(),
            coil_set.get_dl(),
            coil_set.get_r_middle(),
            nn,
            sg,
        )
    )

    len_loss_val = coil_set.get_total_length()
    return B_loss_val + weight_length * len_loss_val
