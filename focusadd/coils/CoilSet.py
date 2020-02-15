

class CoilSet:

	def __init__(self,input_file, ):
		if input_file is not None:
			# READ IN HDF5 FORMAT 
		else:
			# INITIALIZE COILS TO DEFAULT VALUES
		self.compute_r()

	def compute_r(self):


	def get_r(self):
		return self.r

	def get_params(self):
		return self.fc, self.fr

	def write_coils(self, output_file):
		""" Write coils in HDF5 output format"""

	def calc_length(self):


	def compute_frenet(self):
		""" 
		Computes the tangent, normal, and binormal of the axis.

		These functions assume you compute the tangent first, then the normal, 
		then the binormal. 

		"""
		self.compute_tangent()
		self.compute_normal()
		self.compute_binormal()

	def compute_x1y1z1(self):
		x1 = np.zeros(self.N_zeta+1)
		y1 = np.zeros(self.N_zeta+1)
		z1 = np.zeros(self.N_zeta+1)
		for m in range(len(self.xc)):
			arg = m * self.zeta
			x1 += -m * self.xc[m] * np.sin(arg) + m * self.xs[m] * np.cos(arg)
			y1 += -m * self.yc[m] * np.sin(arg) + m * self.ys[m] * np.cos(arg)
			z1 += -m * self.zc[m] * np.sin(arg) + m * self.zs[m] * np.cos(arg)
		self.x1 = x1
		self.y1 = y1 
		self.z1 = z1

	def compute_x2y2z2(self):
		x2 = np.zeros(self.N_zeta+1)
		y2 = np.zeros(self.N_zeta+1)
		z2 = np.zeros(self.N_zeta+1)
		for m in range(len(self.xc)):
			arg = m * self.zeta
			x2 += -m**2 * self.xc[m] * np.cos(arg) - m**2 * self.xs[m] * np.sin(arg)
			y2 += -m**2 * self.yc[m] * np.cos(arg) - m**2 * self.ys[m] * np.sin(arg)
			z2 += -m**2 * self.zc[m] * np.cos(arg) - m**2 * self.zs[m] * np.sin(arg)
		self.x2 = x2
		self.y2 = y2 
		self.z2 = z2

	def compute_x3y3z3(self):
		x3 = np.zeros(self.N_zeta+1)
		y3 = np.zeros(self.N_zeta+1)
		z3 = np.zeros(self.N_zeta+1)
		for m in range(len(self.xc)):
			arg = m * self.zeta
			x3 += m**3 * self.xc[m] * np.sin(arg) - m**3 * self.xs[m] * np.cos(arg)
			y3 += m**3 * self.yc[m] * np.sin(arg) - m**3 * self.ys[m] * np.cos(arg)
			z3 += m**3 * self.zc[m] * np.sin(arg) - m**3 * self.zs[m] * np.cos(arg)
		self.x3 = x3
		self.y3 = y3 
		self.z3 = z

	def compute_torsion(self):
		r1 = self.get_r1()
		r2 = self.get_r2()
		r3 = self.get_r3()
		cross12 = np.cross(r1, r2)
		top = cross12[:,0] * r3[:,0] + cross12[:,1] * r3[:,1] + cross12[:,2] * r3[:,2]
		bottom = np.linalg.norm(cross12,axis=1)
		self.torsion = top / bottom

	def compute_integrated_torsion(self):

	def compute_curvature(self):
		r1 = self.get_r1()
		r2 = self.get_r2()
		cross12 = np.cross(r1, r2)
		top = np.linalg.norm(cross12, axis=1)
		bottom = np.linalg.norm(r1,axis=1)
		self.curvature = top / bottom

	def compute_integrated_curvature(self):

	def compute_dsdz(self):
		x1, y1, z1 = self.x1, self.y1, self.z1
		self.dsdz = np.sqrt(x1**2 + y1**2 + z1**2) # magnitude of first derivative of curve

	def compute_dNdz(self):
		self.dNdz = (- self.curvature[:,np.newaxis] * self.tangent + self.torsion[:,np.newaxis] * self.binormal ) * self.dsdz[:,np.newaxis]

	def compute_dBdz(self):
		self.dBdz = - self.torsion[:,np.newaxis] * self.normal * self.dsdz[:,np.newaxis]

	def compute_tangent(self):
		a0 = self.get_dsdz()
		top = np.concatenate((self.x1[:,np.newaxis],self.y1[:,np.newaxis],self.z1[:,np.newaxis]),axis=1)
		self.tangent = top / a0[:,np.newaxis]

	def compute_normal(self):
		a1 = self.x2 * self.tangent[:,0] + self.y2 * self.tangent[:,1] + self.z2 * self.tangent[:,2]  
		N = np.concatenate((self.x2[:,np.newaxis],self.y2[:,np.newaxis],self.z2[:,np.newaxis]),axis=1) - self.tangent * a1[:,np.newaxis]
		norm = np.linalg.norm(N,axis=1)
		self.normal = N / norm[:,np.newaxis]

	def compute_binormal(self):
		self.binormal = np.cross(self.tangent, self.normal)