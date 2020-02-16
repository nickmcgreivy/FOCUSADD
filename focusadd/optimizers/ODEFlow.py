import jax.numpy as np
from Optimizer import Optimizer
class ODEFlow(Optimizer):

	def __init__(self,loss_function,learning_rate=0.05):
		super().__init__(loss_function)
		self.learning_rate = learning_rate

	def step(self,params):
		"""
		Takes an initial set of parameters and updates the parameters according to a fourth-order Runge-Kutta integrator.

		The equations for the fourth-order integrator are stated in the docs.

		Input: params, a tuple with two fourier series, one for the coils and one for the rotation of the coils.

		Output: (loss, new_params), a tuple with the loss_val and the new parameters 

		"""

		loss_val, g1 = self.value_and_grad(params)
		params_1 = np.subtract(params,-np.multiply(self.learning_rate * 0.5, g1))
		_, g2 = self.value_and_grad(params_1)
		params_2 = np.subtract(params_1,-np.multiply(self.learning_rate * 0.5, g2))
		_, g3 = self.value_and_grad(params_2)
		params_3 = np.subtract(params_2,-np.multiply(self.learning_rate, g3))
		_, g4 = self.value_and_grad(params_4)

		g_sum = np.add(g1, np.multiply(2,g2)) 
		g_sum = np.add(g_sum,np.multiply(2,g3))
		g_sum = np.add(g_sum,g3)
		toSubtract = np.multiply(g_sum,self.learning_rate / 6.)
		new_params = np.subtract(params,toSubtract)
		return loss_val, new_params