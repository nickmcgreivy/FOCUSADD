from jax import grad, value_and_grad, hessian
from jax.ops import index_update, index
import jax.numpy as np
from lossFunctions.LossFunction import LossFunction

class ShapeGradient:

	def __init__(self,surface,coil_set, weight_length = 0.1):
		self.surface = surface
		self.coil_set = coil_set
		self.weight_length = weight_length

	def loss_function(self,r,f_phys=True,f_eng=False,i=None):
		if i is not None:
			coil_r = self.coil_set.get_r_middle()
			coil_r = index_update(coil_r, index[i,:,:,:,:],r)
			B_loss_val = np.sum(LossFunction.bnsquared(self.surface.get_r_central(),\
			self.coil_set.get_I(),self.coil_set.get_dl(),coil_r,\
			self.surface.get_nn(), self.surface.get_sg()))			
		else:
			B_loss_val = np.sum(LossFunction.bnsquared(self.surface.get_r_central(),\
			self.coil_set.get_I(),self.coil_set.get_dl(),r,\
			self.surface.get_nn(), self.surface.get_sg()))

		loss = 0.
		if f_phys:
			loss += B_loss_val
		if f_eng:
			loss += self.weight_length * self.coil_set.get_total_length() # This is broken because it doesn't depend on r, it's already been computed
		return loss


	def coil_gradient(self,f_phys=True,f_eng=False):
		grad_loss = value_and_grad(self.loss_function)
		loss, shape_grad = grad_loss(self.coil_set.get_r_middle(),f_phys=f_phys,f_eng=f_eng)
		#shape_grad = index_update(shape_grad, index[:,0,:,:,:], (shape_grad[:,0,:,:,:] + shape_grad[:,-1,:,:,:]))[:,:-1,:,:,:]
		return loss, shape_grad

	def coil_hessian(self,f_phys=True, f_eng=False,i=0):
		hessian_func = hessian(self.loss_function)
		H = hessian_func(self.coil_set.get_r_middle()[i],f_phys=f_phys,f_eng=f_eng,i=i)
		return H




