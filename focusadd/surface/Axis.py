import jax.numpy as np
import math as m

PI = m.pi

class Axis:

	""" Represents the stellarator magnetic axis. """

	def __init__(self, xc, xs, yc, ys, zc, zs, N_zeta):
		""" Initializes axis from Fourier series, calculates real-space coordinates. """
		self.xc = xc
		self.xs = xs
		self.yc = yc
		self.ys = ys
		self.zc = zc
		self.zs = zs
		self.N_zeta = N_zeta
		self.zeta = np.linspace(0,2*PI, self.N_zeta)
		self.compute_xyz()
		self.compute_x1y1z1()
		self.compute_x2y2z2()
		self.computeFrenet()


	def compute_xyz(self):
		""" From the Fourier harmonics of the axis, computes the real-space coordinates of the axis. """
		x = np.zeros(self.N_zeta)
		y = np.zeros(self.N_zeta)
		z = np.zeros(self.N_zeta)
		for m in range(len(self.xc)):
			arg = m * self.zeta
			x += self.xc[m] * np.cos(arg) + self.xs[m] * np.sin(arg)
			y += self.yc[m] * np.cos(arg) + self.ys[m] * np.sin(arg)
			z += self.zc[m] * np.cos(arg) + self.zs[m] * np.sin(arg)
		self.x = x
		self.y = y
		self.z = z

	def get_xyz(self):
		return self.x, self.y, self.z

	def computeFrenet(self):
		""" 
		Computes the tangent, normal, and binormal of the axis.

		These functions assume you compute the tangent first, then the normal, 
		then the binormal. 

		"""
		self.tangent = self.computeTangent()
		self.normal = self.computeNormal()
		self.binormal = self.computeBinormal()

	def compute_x1y1z1(self):
		x1 = np.zeros(self.N_zeta)
		y1 = np.zeros(self.N_zeta)
		z1 = np.zeros(self.N_zeta)
		for m in range(len(self.xc)):
			arg = m * self.zeta
			x1 += -m * self.xc[m] * np.sin(arg) + m * self.xs[m] * np.cos(arg)
			y1 += -m * self.yc[m] * np.sin(arg) + m * self.ys[m] * np.cos(arg)
			z1 += -m * self.zc[m] * np.sin(arg) + m * self.zs[m] * np.cos(arg)
		self.x1 = x1
		self.y1 = y1 
		self.z1 = z1

	def compute_x2y2z2(self):
		x2 = np.zeros(self.N_zeta)
		y2 = np.zeros(self.N_zeta)
		z2 = np.zeros(self.N_zeta)
		for m in range(len(self.xc)):
			arg = m * self.zeta
			x2 += -m**2 * self.xc[m] * np.cos(arg) - m**2 * self.xs[m] * np.sin(arg)
			y2 += -m**2 * self.yc[m] * np.cos(arg) - m**2 * self.ys[m] * np.sin(arg)
			z2 += -m**2 * self.zc[m] * np.cos(arg) - m**2 * self.zs[m] * np.sin(arg)
		self.x2 = x2
		self.y2 = y2 
		self.z2 = z2

	def compute_x3y3z3(self):
		return NotImplementedError()

	def get_zeta(self):
		return self.zeta

	def get_tangent(self):
		return self.tangent

	def computeTangent(self):
		x1, y1, z1 = self.x1, self.y1, self.z1
		a0 = np.sqrt(x1**2 + y1**2 + z1**2) # magnitude of first derivative of curve
		foo= np.concatenate((x1[:,np.newaxis],y1[:,np.newaxis],z1[:,np.newaxis]),axis=1)
		return foo / a0[:,np.newaxis]

	def get_normal(self):
		return self.normal

	def computeNormal(self):
		x2, y2, z2 = self.x2, self.y2, self.z2
		a1 = x2 * self.tangent[:,0] + y2 * self.tangent[:,1] + z2 * self.tangent[:,2]  
		N = np.concatenate((x2[:,np.newaxis],y2[:,np.newaxis],z2[:,np.newaxis]),axis=1) - self.tangent * a1[:,np.newaxis]
		norm = np.linalg.norm(N,axis=1)
		return N / norm[:,np.newaxis]

	def get_binormal(self):
		return self.binormal

	def computeBinormal(self):
		return np.cross(self.tangent, self.normal)







