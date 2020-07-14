from .LossFunction import LossFunction
import sys
sys.path.append("..")
from coils.CoilSet import CoilSet
import jax.numpy as np
import math
from jax import jit
from functools import partial
from jax.config import config

config.update("jax_enable_x64", True)

PI = math.pi


def default_loss(surface_data, params_to_data, w_args, params, B_extern = None):
	""" 
	Computes the default loss: int (B dot n)^2 dA + weight_length * len(coils) 

	Input: params, a tuple of the fourier series for the coils and a fourier series for the rotation.

	Output: A scalar, which is the loss_val computed by the function. JAX will eventually differentiate
	this in an optimizer.
	"""
	w_B, w_L = w_args
	r_surf, nn, sg = surface_data
	I, dl, r, total_length = params_to_data(params)

	B_loss_val = LossFunction.quadratic_flux(r_surf, I, dl, r, nn, sg, B_extern = B_extern)

	return w_B * B_loss_val + w_L * total_length

def lhd_saddle_B(surface_data, NS):
	""" Only to be called once """
	r_surf, _, _ = surface_data
	fc_s = np.load("initFiles/lhd/lhd_fc_saddle.npy")
	I_s = np.load("initFiles/lhd/lhd_I_saddle.npy")
	theta = np.linspace(0, 2 * PI, NS + 1)
	NF = fc_s.shape[2]
	NC = fc_s.shape[1]
	coil_data = NC, NS, NF, 0, 0, 0, 0, 0, 0, 0
	r_s = CoilSet.compute_r_centroid(coil_data, fc_s, theta)
	dl_s = CoilSet.compute_x1y1z1(coil_data, fc_s, theta) * (2 * PI / NS)
	return LossFunction.biotSavart(r_surf, I_s, dl_s[:,:,None,None,:], r_s[:,:,None,None,:])
