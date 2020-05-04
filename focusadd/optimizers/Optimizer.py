from jax import value_and_grad
from functools import partial
from jax import jit

class Optimizer:

	def __init__(self,loss_function):
		self.loss_function = loss_function
		self.value_and_grad = value_and_grad(loss_function.loss)

