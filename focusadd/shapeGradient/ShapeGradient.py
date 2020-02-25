from jax import grad
import jax.numpy as np
from lossFunctions.LossFunction import LossFunction

class ShapeGradient:

	def __init__(self,surface,coil_set, weight_length = 0.1):
		self.surface = surface
		self.coil_set = coil_set
		self.weight_length = weight_length



	def loss_function(self,r):
		dl = (r[:,1:,:,:,:] - r[:,:-1,:,:,:])
		r_middle = (r[:,1:,:,:,:] + r[:,:-1,:,:,:]) / 2.
		B_loss_val = np.sum(LossFunction.bnsquared(self.surface.get_r_central(),\
			self.coil_set.get_I(),dl,r_middle,\
			self.surface.get_nn(), self.surface.get_sg(), self.surface.NT, self.surface.NZ,\
			 self.coil_set.NNR, self.coil_set.NBR))
		len_loss_val = np.sum(np.linalg.norm(dl,axis=-1))
		return B_loss_val + self.weight_length * len_loss_val




	def coil_gradient(self):
		grad_loss = grad(self.loss_function)
		return grad_loss(self.coil_set.get_r())




