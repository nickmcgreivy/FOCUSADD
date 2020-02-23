from lossFunctions.LossFunction import LossFunction
import jax.numpy as np
class DefaultLoss(LossFunction):

	def __init__(self,surface,coil_set,weight_length=0.1):
		super().__init__(surface,coil_set)
		self.weight_length = 0.1

	def loss(self,params):
		""" 
		Computes the default loss: int (B dot n)^2 dA + weight_length * len(coils) 

		Input: params, a tuple of the fourier series for the coils and a fourier series for the rotation.

		Output: A scalar, which is the loss_val computed by the function. JAX will eventually differentiate
		this in an optimizer.
		"""

		# NEED TO SET_PARAMS 
		self.coil_set.set_params(params) 

		B_loss_val = np.sum(self.bnsquared())

		len_loss_val = self.coil_set.get_total_length()
		print(self.weight_length * len_loss_val)
		print(B_loss_val)

		return B_loss_val + self.weight_length * len_loss_val

