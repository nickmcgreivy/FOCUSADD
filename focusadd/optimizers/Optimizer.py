from jax import value_and_grad

class Optimizer:

	def __init__(self,loss_function):
		self.loss_function = loss_function
		self.value_and_grad = value_and_grad(self.loss_function.loss)

