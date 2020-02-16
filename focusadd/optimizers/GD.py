import jax.numpy as np
from Optimizer import Optimizer

class GD(Optimizer):

	def __init__(self,loss_function,learning_rate=0.05):
		super().__init__(loss_function)
		self.learning_rate = learning_rate

	def step(self,params):
		"""
		Takes an initial set of parameters and updates the parameters according to gradient descent.

		Input: params, a tuple with two fourier series, one for the coils and one for the rotation of the coils.

		Output: (loss, new_params), a tuple with the loss_val and the new parameters

		"""

		loss_val, grad = self.value_and_grad(params)

		new_params = np.subtract(params,np.multiply(self.learning_rate * grad))
		return loss_val, new_params

