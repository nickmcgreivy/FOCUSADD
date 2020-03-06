from jax import grad, value_and_grad, hessian
from jax.ops import index_update, index
import jax.numpy as np
from lossFunctions.LossFunction import LossFunction

class ShapeGradient:

	def __init__(self,surface,coil_set, weight_length = 0.1):
		self.surface = surface
		self.coil_set = coil_set
		self.weight_length = weight_length

	def loss_function(self,r,f_phys=True,f_eng=True):
		#dl = (r[:,1:,:,:,:] - r[:,:-1,:,:,:])
		dl = np.zeros(r.shape)
		dl = index_update(dl, index[:,:-1,:,:,:], r[:,1:,:,:,:] - r[:,:-1,:,:,:])
		dl = index_update(dl, index[:,-1,:,:,:], r[:,0,:,:,:] - r[:,-1,:,:,:])
		r_middle = (r[:,1:,:,:,:] + r[:,:-1,:,:,:]) / 2.
		r_middle = np.zeros(r.shape)
		r_middle = index_update(r_middle, index[:,:-1,:,:,:], (r[:,:-1,:,:,:] + r[:,1:,:,:,:])/2)
		r_middle = index_update(r_middle, index[:,-1,:,:,:] , (r[:,-1,:,:,:] + r[:,0,:,:,:])/2)

		B_loss_val = np.sum(LossFunction.bnsquared(self.surface.get_r_central(),\
			self.coil_set.get_I(),dl,r_middle,\
			self.surface.get_nn(), self.surface.get_sg(), self.surface.NT, self.surface.NZ,\
			 self.coil_set.NNR, self.coil_set.NBR))
		len_loss_val = self.coil_set.get_total_length()
		loss = 0.
		if f_phys:
			loss += B_loss_val
		if f_eng:
			loss += self.weight_length * len_loss_val
		return loss


	def coil_gradient(self,f_phys=True,f_eng=True):
		grad_loss = value_and_grad(self.loss_function)
		loss, shape_grad = grad_loss(self.coil_set.get_r()[:,:-1,:,:,:],f_phys=f_phys,f_eng=f_eng)
		#shape_grad = index_update(shape_grad, index[:,0,:,:,:], (shape_grad[:,0,:,:,:] + shape_grad[:,-1,:,:,:]))[:,:-1,:,:,:]
		return loss, shape_grad

	def coil_hessian(self,f_phys=True, f_eng=True):
		hessian_func = hessian(self.loss_function)
		H = hessian_func(self.coil_set.get_r()[:,:-1,:,:,:],f_phys=f_phys,f_eng=f_eng)
		return H




