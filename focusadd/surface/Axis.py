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
		self.x, self.y, self.z = self.compute_xyz()


	def compute_xyz(self):
		""" From the Fourier harmonics of the axis, computes the real-space coordinates of the axis. """
		x = np.zeros(self.N_zeta)
		y = np.zeros(self.N_zeta)
		z = np.zeros(self.N_zeta)
		zeta = np.linspace(0,2*PI, self.N_zeta)
		for m in range(len(self.xc)):
			arg = m * zeta
			x += self.xc[m] * np.cos(arg) + self.xs[m] * np.sin(arg)
			y += self.yc[m] * np.cos(arg) + self.ys[m] * np.sin(arg)
			z += self.zc[m] * np.cos(arg) + self.zs[m] * np.sin(arg)
		return x, y, z

	def get_xyz(self):
		return self.x, self.y, self.z

	def get_zeta(self):
		return np.linspace(0,2*PI, self.N_zeta)




