from .Axis import Axis
import numpy as np
import math as m

PI = m.pi

class Surface:

	def __init__(self, axis, num_zeta, num_theta, epsilon, minor_rad, N_rotate, zeta_off,s):
		self.axis = axis
		self.NT = num_theta
		self.NZ = num_zeta
		self.epsilon = epsilon
		self.a = minor_rad
		self.NR = N_rotate
		self.zeta_off = zeta_off
		self.s = s
		self.initialize_surface()

	def initialize_surface(self):
		self.calc_frame()
		self.calc_r()
		self.calc_nn()

	def calc_alpha(self):
		torsion = self.axis.get_torsion()
		d_zeta = 2. * PI / self.NZ
		torsionInt = (np.cumsum(torsion) - torsion) * d_zeta
		zeta = self.axis.get_zeta()
		alpha = 0.5 * self.NR * zeta + self.zeta_off
		alpha -= torsionInt
		self.alpha = alpha

	def get_alpha(self):
		return self.alpha

	def calc_frame(self):
		self.calc_alpha()
		calpha = np.cos(self.alpha)
		salpha = np.sin(self.alpha)
		N = self.axis.get_normal()
		B = self.axis.get_binormal()
		self.v1 = calpha[:,np.newaxis] * N + salpha[:,np.newaxis] * B
		self.v2 = -salpha[:,np.newaxis] * N + calpha[:,np.newaxis] * B

	def get_frame(self):
		return self.v1, self.v2

	def calc_r(self):
		r = np.zeros((self.NZ+1,self.NT+1,3))
		sa = self.s * self.a
		zeta = self.axis.get_zeta()
		theta = np.linspace(0.,2.*PI,self.NT+1)
		ctheta = np.cos(theta)
		stheta = np.sin(theta)
		ep = self.epsilon
		r += self.axis.get_r()[:,np.newaxis,:]
		r += sa * np.sqrt(ep) * self.v1[:,np.newaxis,:] * ctheta[np.newaxis,:,np.newaxis]
		r += sa * self.v2[:,np.newaxis,:] * stheta[np.newaxis,:,np.newaxis] / np.sqrt(ep)
		self.r = r
		self.r_central = (self.r[1:,1:,:] + self.r[1:,:-1,:] + self.r[:-1,1:,:] + self.r[:-1,:-1,:]) / 4.


	def calc_r_coils(self,num_coils,num_segments,coil_radius):
		r = np.zeros((num_coils,num_segments+1,3))
		sa = coil_radius * self.a
		spacing = int(self.NZ/num_coils)
		zeta = self.axis.get_zeta()[0:self.NZ:spacing]
		theta = np.linspace(0.,2.*PI,num_segments+1)
		ctheta = np.cos(theta)
		stheta = np.sin(theta)
		ep = self.epsilon
		r += self.axis.get_r()[0:self.NZ:spacing,np.newaxis,:]
		r += sa * np.sqrt(ep) * self.v1[0:self.NZ:spacing,np.newaxis,:] * ctheta[np.newaxis,:,np.newaxis]
		r += sa * self.v2[0:self.NZ:spacing,np.newaxis,:] * stheta[np.newaxis,:,np.newaxis] / np.sqrt(ep)
		return r
		

	def calc_drdt(self):
		drdt = np.zeros((self.NZ+1,self.NT+1,3))
		s=1.
		sa = s * self.a
		zeta = self.axis.get_zeta()
		theta = np.linspace(0.,2.*PI,self.NT+1)
		ctheta = np.cos(theta)
		stheta = np.sin(theta)
		ep = self.epsilon
		drdt -= sa * np.sqrt(ep) * self.v1[:,np.newaxis,:] * stheta[np.newaxis,:,np.newaxis]
		drdt += sa * self.v2[:,np.newaxis,:] * ctheta[np.newaxis,:,np.newaxis] / np.sqrt(ep)
		self.drdt = drdt


	def get_drdt(self):
		return self.drdt

	def calc_drdz(self):
		drdz = np.zeros((self.NZ+1,self.NT+1,3))
		s=1.
		sa = s * self.a
		zeta = self.axis.get_zeta()
		theta = np.linspace(0.,2.*PI,self.NT+1)
		ctheta = np.cos(theta)
		stheta = np.sin(theta)
		ep = self.epsilon
		drdz += self.axis.get_r1()[:,np.newaxis,:]
		calpha = np.cos(self.alpha)
		salpha = np.sin(self.alpha)
		dNdz = self.axis.get_dNdz()
		dBdz = self.axis.get_dBdz()
		dalphadz = self.NR / 2. - self.axis.get_torsion()
		dv1dz = calpha[:,np.newaxis] * dNdz \
				+ salpha[:,np.newaxis] * dBdz \
				- self.axis.get_normal() * salpha[:,np.newaxis] * dalphadz[:,np.newaxis] \
				+ calpha[:,np.newaxis] * dalphadz[:,np.newaxis] * self.axis.get_binormal()
		dv2dz = - salpha[:,np.newaxis] * dNdz \
				+ calpha[:,np.newaxis] * dBdz \
				- calpha[:,np.newaxis] * dalphadz[:,np.newaxis] * self.axis.get_normal() \
				- salpha[:,np.newaxis] * dalphadz[:,np.newaxis] * self.axis.get_binormal()

		drdz += sa * np.sqrt(ep) * dv1dz[:,np.newaxis,:] * ctheta[np.newaxis,:,np.newaxis]
		drdz += sa * dv2dz[:,np.newaxis,:] * stheta[np.newaxis,:,np.newaxis] / np.sqrt(ep)
		self.drdz = drdz


	def get_drdz(self):
		return self.drdz

	def get_r(self):
		return self.r

	def get_r_central(self):
		return self.r_central

	def calc_nn(self):
		self.calc_drdt()
		self.calc_drdz()
		nn = np.cross(self.drdt, self.drdz)
		self.sg = np.linalg.norm(nn,axis=2)
		self.nn = nn / self.sg[:,:,np.newaxis]
		self.nn_central = (self.nn[1:,1:,:] + self.nn[1:,:-1,:] + self.nn[:-1,1:,:] + self.nn[:-1,:-1,:]) / 4.

	def get_nn(self):
		return self.nn

	def get_nn_central(self):
		return self.nn_central

	def get_sg(self):
		return self.sg

