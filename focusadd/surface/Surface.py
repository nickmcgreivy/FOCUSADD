import Axis as Axis
import numpy as np

class Surface:

	def __init__(self, axis, num_zeta, num_theta, epsilon, minor_rad, N_rotate, zeta_off):
		self.axis = axis
		self.NT = num_theta
		self.NZ = num_zeta
		self.epsilon = epsilon
		self.minor_rad = minor_rad
		self.N_rotate = N_rotate
		self.zeta_off = zeta_off
		self.initialize_surface()

	def initialize_surface(self):
		self.calc_frame()
		self.calc_xx()
		self.calc_nn()
		self.calc_sg()

	def calc_alpha(self):

	def get_alpha(self):
		return self.alpha


	def calc_frame(self):
		self.calc_alpha()


	def get_frame(self):
		return self.v1, self.v2

	def calc_xx(self):

	def get_xx(self):
		return self.xx

	def calc_nn(self):

	def get_nn(self):
		return self.nn

	def calc_sg(self):

	def get_sg(self):
		return self.sg

