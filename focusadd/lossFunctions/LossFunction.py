import jax.numpy as np
from jax import jit
import math
from memory_profiler import profile
PI = math.pi
class LossFunction:

	def __init__(self,surface,coil_set):
		self.surface = surface
		self.coil_set = coil_set

	def bnsquared(r, I, dl, l, nn, sg):
		""" 

		Computes 1/2(B dot n)^2 dA over the surface from the coils, but doesn't sum over the entire surface yet. 
			
		Requires: A CoilSet and a Surface, as method variables.
		
		Returns: A N_zeta by N_theta array which computes 1/2(B dot n)^2 dA at each point on the surface. 
		We can eventually sum over this array to get the total integral over the surface. 

		"""
		mu_0 = 1.
		mu_0I = I * mu_0
		mu_0Idl = mu_0I[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis] * dl # NC x NS x NNR x NBR x 3
		r_minus_l = r[np.newaxis,:,:,np.newaxis,np.newaxis,np.newaxis,:] - l[:,np.newaxis,np.newaxis,:,:,:,:] # NC x NZ x NT x NS x NNR x NBR x 3
		top = np.cross(mu_0Idl[:,np.newaxis,np.newaxis, :,:,:,:],r_minus_l) # NC x NZ x NT x NS x NNR x NBR x 3
		bottom = np.linalg.norm(r_minus_l,axis=-1)**3 # NC x NZ x NT x NS x NNR x NBR
		B = np.sum(top / bottom[:,:,:,:,:,:,np.newaxis], axis=(0,3,4,5)) # NZ x NT x 3
		return .5 * np.sum(nn * B, axis=-1)**2 * sg #NZ x NT 



