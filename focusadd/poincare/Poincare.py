import numpy as np
import math

PI = math.pi

class Poincare():

	def __init__(self,coil_set,surface):
		self.coil_set = coil_set
		self.surface = surface
		self.axis = self.surface.get_axis()
		self.xList = []
		self.yList = []
		self.zList = []

	def computeB(self,r,theta,z):
		"""
			Inputs:

			r, theta, z : The coordinates of the point we want the magnetic field at. Cylindrical coordinates.

			Outputs: 

			B_z, B_theta, B_z : the magnetic field components at the input coordinates created by the currents in the coils. Cylindrical coordinates.
		"""
		x = r * np.cos(theta)
		y = r * np.sin(theta)
		xyz = np.asarray([x,y,z])
		l = self.coil_set.get_r_middle()
		I = self.coil_set.get_I()
		NNR = self.coil_set.NNR
		NBR = self.coil_set.NBR
		dl = self.coil_set.get_dl()
		mu_0 = 1. 
		mu_0I = I * mu_0 # NC
		mu_0Idl = mu_0I[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis] * dl # NC x NS x NNR x NBR x 3
		r_minus_l = xyz[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:] - l[:,:,:,:,:] # NC x NS x NNR x NBR x 3
		top = np.cross(mu_0Idl,r_minus_l) # NC x x NS x NNR x NBR x 3
		bottom = np.linalg.norm(r_minus_l,axis=-1)**3 # NC x NS x NNR x NBR
		B_xyz = np.sum(top / bottom[:,:,:,:,np.newaxis], axis=(0,1,2,3)) # 3, xyz coordinates
		B_x = B_xyz[0]
		B_y = B_xyz[1]
		B_z = B_xyz[2]
		B_r = B_x * np.cos(theta) + B_y * np.sin(theta)
		B_theta = - B_x * np.sin(theta) + B_y * np.cos(theta)
		return B_r, B_theta, B_z

	def odeStep(self,r,theta,z):
		"""
			Uses the 4th-order Runge-Kutta integration scheme to integrate along the ODEs
			describing the magnetic field. 

			Inputs:

			r, theta, z: The coordinates of the point we begin at during our integration.
			dtheta: The step size in theta for the integration.

			Outputs:

			Fr : dr/dtheta, the change in the r-position of a field line per change in theta
			Fz : dz/dtheta, the change in the z-position of a field line per change in theta
		"""
		B_r, B_theta, B_z = self.computeB(r, theta, z) 
		Fr = (B_r * r / B_theta)
		Fz = (B_z * r / B_theta)
		return Fr, Fz


	def rkStep(self, r_i, theta_i, z_i,dtheta):
		""" 
		A single Runge-Kutta fourth order step 

		Inputs

		r_i , theta_i, z_i : The initial coordinates of the magnetic field line
		dtheta : the step size of the Runge-Kutta step

		Returns

		r_f, theta_f, z_f : The final coordinates of the magnetic field line after
		the Runge-Kutta step


		""" 
		Fr_1, Fz_1 = self.odeStep(r_i,theta_i,z_i)
		r_2 = r_i + (dtheta/2.) * Fr_1
		z_2 = z_i + (dtheta/2.) * Fz_1

		Fr_2, Fz_2 = self.odeStep(r_2,theta_i + dtheta/2.,z_2)
		r_3 = r_i + (dtheta/2.) * Fr_2
		z_3 = z_i + (dtheta/2.) * Fz_2

		Fr_3, Fz_3 = self.odeStep(r_3, theta_i + dtheta/2., z_3)
		r_4 = r_i + dtheta * Fr_3
		z_4 = z_i + dtheta * Fz_3

		Fr_4, Fz_4 = self.odeStep(r_4, theta_i + dtheta, z_4)



		r_f = r_i + dtheta * (Fr_1 + 2. * Fr_2 + 2. * Fr_3 + Fr_4) / 6. 
		theta_f = theta_i + dtheta
		z_f = z_i + dtheta * (Fz_1 + 2. * Fz_2 + 2. * Fz_3 + Fz_4) / 6.
		return r_f, theta_f, z_f


	# Will want to parallelize
	def odeIntegrate(self,r_i,theta_i,z_i,N_step,N_poincare):
		"""
			Integrates the magnetic field line flow around the torus for a single field line
			to get a set of Poincare points for that field line. 

			Inputs:

			r_i, theta_i, z_i: The initial coordinate of the field line
			N_step: The number of Runge-Kutta steps per transit around the torus; dtheta = 2pi/N_step
			N_poincare: The number of rotations around the torus, which gives the number of Poincare points

			Outputs:

			x, y, z: Lists of length N_poincare of x, y, and z points where the magnetic field line crosses the initial theta. 
		"""
		#rs = np.zeros(N_poincare*N_step+1)
		#thetas = np.zeros(N_poincare*N_step+1)
		#zs = np.zeros(N_poincare*N_step+1)
		#rs =np.zeros(N_poincare+1) 
		#thetas = np.zeros(N_poincare+1)
		#zs = np.zeros(N_poincare+1)
		rs = []
		thetas = []
		zs = []

		rs.append(r_i)
		thetas.append(theta_i)
		zs.append(z_i)
		dtheta = 2. * PI / N_step

		r = r_i
		theta = theta_i
		z = z_i

		for i in range(N_poincare):
			for j in range(N_step):
				r, theta, z = self.rkStep(r,theta,z,dtheta)
			theta = theta % (2 * PI)
			rs.append(r)
			thetas.append(theta)
			zs.append(z)
		return rs, thetas, zs







	def getPoincarePoints(self,N_step,N_poincare,theta,radii):
		"""

			NOTE: THIS ONLY WORKS FOR THETA = 0 RIGHT NOW

			Loop over N_poincare points in the plot, equally spaced in toroidal
			radius starting from the magnetic axis at a given theta, and 
			creating a list of points where the plot punctures.

			Inputs:

			N_step : number of RK steps per poincare transit, dtheta = 2pi/N_step
			N_poincare : number of times we puncture the poincare plot, starting at radius r
			theta : Which theta will our poincare plot track? We are getting a poloidal cross-section at theta.
			radii : list of radii r which we start at, away from our magnetic axis. We expect this to range from 0 
			to slightly higher than 1. 

			Outputs:

			rs, thetas, zs : lists of floating point numbers where the field lines intersect at a given theta. 

		"""

		x_axis,y_axis,z_axis = self.axis.get_r_from_zeta(theta)
		r_axis = np.sqrt(x_axis ** 2 + y_axis ** 2)
		v1, _ = self.axis.get_frame() # v1, v2
		v1 = v1[0,:] # at zeta = 0 THIS LINE BREAKS THE CODE
		ep = self.axis.epsilon
		saep = self.axis.a * self.surface.s * np.sqrt(ep)
		ct = np.cos(theta)
		st = np.sin(theta)
		rs = []
		thetas = []
		zs = []
		for r in radii:
			v1_normalized = r * saep * v1
			r_i = r_axis + ct * v1_normalized[0] + st * v1_normalized[1]
			# theta = theta
			z_i = z_axis + v1[2]
			rlist, thetalist, zlist = self.odeIntegrate(r_i, theta, z_i, N_step, N_poincare)
			rs = rs + rlist
			thetas = thetas + thetalist
			zs = zs + zlist
		return rs, thetas, zs





	

