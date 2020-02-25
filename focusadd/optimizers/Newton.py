import jax.numpy as np
from optimizers.Optimizer import Optimizer
from jax import hessian

class Newton(Optimizer):

	def __init__(self,loss_function,damping_rate=0.5,diag_add=1.0):
		super().__init__(loss_function)
		self.damping_rate = damping_rate
		self.hessian = hessian(self.loss_function.loss)

	def step(self,params):
		"""
		Takes an initial set of parameters and updates the parameters according to a damped Newton update.

		Input: params, a tuple with two fourier series, one for the coils and one for the rotation of the coils.

		Output: (loss, new_params), a tuple with the loss_val and the new parameters

		"""
		"""loss_val, g = self.value_and_grad(params)
		H = self.hessian(params)
		print(type(H))
		print(H)
		print(np.linalg.eig(H))
		return loss_val,params"""
		pass