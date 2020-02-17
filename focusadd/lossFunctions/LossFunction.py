class LossFunction:

	def __init__(self,surface,coil_set):
		self.surface = surface
		self.coil_set = coil_set


	def bnsquared(self):
		""" 

		Computes (B dot n)^2 over the surface from the coils. 
			
		Requires: A CoilSet and a Surface, as method variables.
		
		Returns: A N_zeta by N_theta array which computes (B dot n)^2 at each point on the surface. 
		We can eventually sum over this array to get the total integral over the surface. 

		"""
		mu_0 = 1.
		r = self.surface.get_r() # NZ+1 x NT+1 x 3
		r = (r[1:,1:,:] + r[1:,:-1,:] + r[:-1,1:,:] + r[:-1,:-1,:]) / 4. # NZ x NT x 3
		mu_0I = self.coil_set.get_I() * mu_0 / (self.coil_set.NNR * self.coil_set.NBR) # NC
		r_prime = self.coil_set.get_r() # NC x NT+1 x NNR x NBR x 3
		r_prime = (r_prime[:,1:,:,:,:] + r_prime[:,:-1,:,:,:]) / 2. # NC x NT x NNR x NBR x 3
		mu_0Idl = mu_0I[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis] * (r_prime[:,1:,:,:,:] - r_prime[:,:-1,:,:,:]) / 2. # NC x NT x NNR x NBR x 3
		r_minus_rprime = r[np.newaxis,:,:,np.newaxis,np.newaxis,:] - r_prime[:,np.newaxis,:,:,:,:] # NC x NZ x NT x NNR x NBR x 3
		top = np.cross(mu_0Idl[:,np.newaxis,:,:,:,:],r_minus_rprime)
		bottom = np.linalg.norm(r_minus_rprime,axis=-1)**3

		toSum = top / bottom[:,:,:,:,:,np.newaxis]
		B = np.sum(toSum,axis=(0,3,4))
		normal = self.surface.

