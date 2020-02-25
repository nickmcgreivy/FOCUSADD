from lossFunctions.LossFunction import LossFunction
import jax.numpy as np
import math

PI = math.pi 
class DefaultLoss(LossFunction):

	def __init__(self,surface,coil_set,weight_length=0.1):
		super().__init__(surface,coil_set)
		self.weight_length = weight_length

	def loss(self,params):
		""" 
		Computes the default loss: int (B dot n)^2 dA + weight_length * len(coils) 

		Input: params, a tuple of the fourier series for the coils and a fourier series for the rotation.

		Output: A scalar, which is the loss_val computed by the function. JAX will eventually differentiate
		this in an optimizer.
		"""

		# NEED TO SET_PARAMS 
		self.coil_set.set_params(params) 
		B_loss_val = np.sum(LossFunction.bnsquared(self.surface.get_r_central(),\
			self.coil_set.get_I(),self.coil_set.get_dl(),self.coil_set.get_r_middle(),\
			self.surface.get_nn(), self.surface.get_sg(), self.surface.NT, self.surface.NZ,\
			 self.coil_set.NNR, self.coil_set.NBR))

		len_loss_val = self.coil_set.get_total_length()
		return B_loss_val + self.weight_length * len_loss_val

