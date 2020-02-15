

class CoilSet:

	def __init__(self,num_coils, num_segments, num_fourier_coils):
		self.NC = num_coils
		self.NS = num_segments
		self.fc = {} # central coil fourier series
		self.fr = {} # rotation fourier series


	
	def get_r(self):
		return self.r

	def get_params(self):
		return self.fc, self.fr