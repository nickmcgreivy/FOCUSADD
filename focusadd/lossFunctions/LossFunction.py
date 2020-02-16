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
		pass

