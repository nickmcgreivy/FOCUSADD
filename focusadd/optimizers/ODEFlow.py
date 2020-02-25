import jax.numpy as np
from optimizers.Optimizer import Optimizer
from jax import jit
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
		fc, fr = params
		loss_val, g1 = self.value_and_grad(params)
		gfc1, gfr1 = g1
		params_1 = (fc - self.learning_rate * 0.5 * gfc1, fr - self.learning_rate * 0.5 * gfr1)
		_, g2 = self.value_and_grad(params_1)
		gfc2, gfr2 = g2
		fc1, fr1 = params_1
		params_2 = (fc1 - self.learning_rate * 0.5 * gfc2, fr1 - self.learning_rate * 0.5 * gfr2)
		_, g3 = self.value_and_grad(params_2)
		gfc3, gfr3 = g3
		fc2, fr2 = params_2
		params_3 = (fc1 - self.learning_rate * 0.5 * gfc3, fr2 - self.learning_rate * gfr3)
		_, g4 = self.value_and_grad(params_3)
		gfc4, gfr4 = g4
		g_sum = ((gfc1 + 2. * gfc2 + 2. * gfc3 + gfc4) * self.learning_rate / 6., (gfr1 + 2. * gfr2 + 2. * gfr3 + gfr4) * self.learning_rate / 6.)
		g_sum_c, g_sum_r = g_sum
		new_params = (fc - g_sum_c, fr - g_sum_r)
		return loss_val, new_params