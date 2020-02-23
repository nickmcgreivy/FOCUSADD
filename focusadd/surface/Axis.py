import numpy as np
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
		self.NF = len(self.xc)
		self.N_zeta = N_zeta
		self.zeta = np.linspace(0,2*PI, self.N_zeta+1)
		self.compute_xyz()
		self.compute_x1y1z1()
		self.compute_x2y2z2()
		self.compute_x3y3z3()
		self.compute_dsdz()
		self.compute_frenet()
		self.compute_torsion()
		self.compute_curvature()
		self.compute_dBdz()
		self.compute_dNdz()


	def compute_xyz(self):
		""" From the Fourier harmonics of the axis, computes the real-space coordinates of the axis. """
		x = np.zeros(self.N_zeta+1)
		y = np.zeros(self.N_zeta+1)
		z = np.zeros(self.N_zeta+1)
		for m in range(self.NF):
			arg = m * self.zeta
			x += self.xc[m] * np.cos(arg) + self.xs[m] * np.sin(arg)
			y += self.yc[m] * np.cos(arg) + self.ys[m] * np.sin(arg)
			z += self.zc[m] * np.cos(arg) + self.zs[m] * np.sin(arg)
		self.x = x
		self.y = y
		self.z = z

	def get_xyz(self):
		return self.x, self.y, self.z

	def get_r(self):
		return np.concatenate((self.x[:,np.newaxis],self.y[:,np.newaxis],self.z[:,np.newaxis]),axis=1)

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
		for m in range(self.NF):
			arg = m * self.zeta
			x1 += -m * self.xc[m] * np.sin(arg) + m * self.xs[m] * np.cos(arg)
			y1 += -m * self.yc[m] * np.sin(arg) + m * self.ys[m] * np.cos(arg)
			z1 += -m * self.zc[m] * np.sin(arg) + m * self.zs[m] * np.cos(arg)
		self.x1 = x1
		self.y1 = y1 
		self.z1 = z1

	def get_r1(self):
		return np.concatenate((self.x1[:,np.newaxis],self.y1[:,np.newaxis],self.z1[:,np.newaxis]),axis=1)

	def compute_x2y2z2(self):
		x2 = np.zeros(self.N_zeta+1)
		y2 = np.zeros(self.N_zeta+1)
		z2 = np.zeros(self.N_zeta+1)
		for m in range(self.NF):
			arg = m * self.zeta
			x2 += -m**2 * self.xc[m] * np.cos(arg) - m**2 * self.xs[m] * np.sin(arg)
			y2 += -m**2 * self.yc[m] * np.cos(arg) - m**2 * self.ys[m] * np.sin(arg)
			z2 += -m**2 * self.zc[m] * np.cos(arg) - m**2 * self.zs[m] * np.sin(arg)
		self.x2 = x2
		self.y2 = y2 
		self.z2 = z2

	def get_r2(self):
		return np.concatenate((self.x2[:,np.newaxis],self.y2[:,np.newaxis],self.z2[:,np.newaxis]),axis=1)

	def compute_x3y3z3(self):
		x3 = np.zeros(self.N_zeta+1)
		y3 = np.zeros(self.N_zeta+1)
		z3 = np.zeros(self.N_zeta+1)
		for m in range(self.NF):
			arg = m * self.zeta
			x3 += m**3 * self.xc[m] * np.sin(arg) - m**3 * self.xs[m] * np.cos(arg)
			y3 += m**3 * self.yc[m] * np.sin(arg) - m**3 * self.ys[m] * np.cos(arg)
			z3 += m**3 * self.zc[m] * np.sin(arg) - m**3 * self.zs[m] * np.cos(arg)
		self.x3 = x3
		self.y3 = y3 
		self.z3 = z3

	def get_r3(self):
		return np.concatenate((self.x3[:,np.newaxis],self.y3[:,np.newaxis],self.z3[:,np.newaxis]),axis=1)

	def get_zeta(self):
		return self.zeta

	def get_tangent(self):
		return self.tangent

	def compute_tangent(self):
		a0 = self.get_dsdz()
		top = self.get_r1()
		self.tangent = top / a0[:,np.newaxis]

	def get_normal(self):
		return self.normal

	def compute_normal(self):
		a1 = self.x2 * self.tangent[:,0] + self.y2 * self.tangent[:,1] + self.z2 * self.tangent[:,2]
		N = self.get_r2() - self.tangent * a1[:,np.newaxis]
		norm = np.linalg.norm(N,axis=1)
		self.normal = N / norm[:,np.newaxis]

	def get_binormal(self):
		return self.binormal

	def compute_binormal(self):
		self.binormal = np.cross(self.tangent, self.normal)

	def compute_torsion(self):
		r1 = self.get_r1()
		r2 = self.get_r2()
		r3 = self.get_r3()
		cross12 = np.cross(r1, r2)
		top = cross12[:,0] * r3[:,0] + cross12[:,1] * r3[:,1] + cross12[:,2] * r3[:,2]
		bottom = np.linalg.norm(cross12,axis=1)**2
		self.torsion = top / bottom

	def get_torsion(self):
		return self.torsion

	def compute_curvature(self):
		r1 = self.get_r1()
		r2 = self.get_r2()
		cross12 = np.cross(r1, r2)
		top = np.linalg.norm(cross12, axis=1)
		bottom = np.linalg.norm(r1,axis=1)**3
		self.curvature = top / bottom

	def get_curvature(self):
		return self.curvature

	def compute_dsdz(self):
		x1, y1, z1 = self.x1, self.y1, self.z1
		self.dsdz = np.sqrt(x1**2 + y1**2 + z1**2) # magnitude of first derivative of curve

	def get_dsdz(self):
		return self.dsdz

	def compute_dNdz(self):
		self.dNdz = (- self.curvature[:,np.newaxis] * self.tangent + self.torsion[:,np.newaxis] * self.binormal ) * self.dsdz[:,np.newaxis]

	def get_dNdz(self):
		return self.dNdz

	def compute_dBdz(self):
		self.dBdz = - self.torsion[:,np.newaxis] * self.normal * self.dsdz[:,np.newaxis]

	def get_dBdz(self):
		return self.dBdz





