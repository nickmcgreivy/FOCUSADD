import jax.numpy as np
from jax.experimental.ode import odeint
from functools import partial
from jax.config import config
import jax
from jax import jit
import matplotlib.pyplot as plt
import sys
import time
sys.path.append("..")

from surface.Surface import Surface
from coils.CoilSet import CoilSet
config.update("jax_enable_x64", True)


PI = np.pi

class Poincare():

	def computeB(I, dl, l, r, theta, z):
		"""
			Inputs:

			r, theta, z : The coordinates of the point we want the magnetic field at. Cylindrical coordinates.

			Outputs: 

			B_z, B_theta, B_z : the magnetic field components at the input coordinates created by the currents in the coils. Cylindrical coordinates.
		"""
		x = r * np.cos(theta)
		y = r * np.sin(theta)
		xyz = np.asarray([x,y,z])
		
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

	def step(B_func, arr, theta):
		r = arr[0]
		z = arr[1]
		B_r, B_theta, B_z = B_func(r, theta, z) 
		Fr = (B_r * r / B_theta)
		Fz = (B_z * r / B_theta)
		return np.asarray((Fr, Fz))


	def getPoincarePoints(N_poincare, theta, radii, surface, is_frenet, coil_data, coil_params):
		"""

			NOTE: THIS ONLY WORKS FOR THETA = 0 RIGHT NOW

			Loop over N_poincare points in the plot, equally spaced in toroidal
			radius starting from the magnetic axis at a given theta, and 
			creating a list of points where the plot punctures.

			Inputs:

			N_poincare : number of times we puncture the poincare plot, starting at radius r
			theta : Which theta will our poincare plot track? We are getting a poloidal cross-section at theta.
			radii : list of radii r which we start at, away from our magnetic axis. We expect this to range from 0 
			to slightly higher than 1. 

			Outputs:

			rs, zs : lists of floating point numbers where the field lines intersect at a given theta. 

		"""

		axis = surface.get_axis()
		x_axis,y_axis,z_axis = axis.get_r_from_zeta(theta)
		r_axis = np.sqrt(x_axis ** 2 + y_axis ** 2)
		v1, _ = axis.get_frame()
		v1 = v1[0, :] # at zeta = 0 THIS LINE BREAKS THE CODE
		ep = axis.epsilon
		saep = axis.a * surface.s * np.sqrt(ep)
		ct = np.cos(theta)
		st = np.sin(theta)

		rs = np.asarray([])
		zs = np.asarray([])
		theta_i_f = [0, 2 * PI * N_poincare] # THIS LINE DOES TOO
		I, dl, _, l, _ = CoilSet.get_outputs(coil_data, is_frenet, coil_params)
		B_func = partial(Poincare.computeB, I, dl, l)
		step_partial = jit(partial(Poincare.step, B_func))
		t_eval = np.linspace(0, 2 * PI * N_poincare, N_poincare + 1)

		@jit
		def update(r):
			v1_normalized = r * saep * v1
			y = np.asarray((r_axis + ct * v1_normalized[0] + st * v1_normalized[1], z_axis + v1_normalized[2]))
			sol = odeint(step_partial, y, t_eval)
			return sol[:, 0], sol[:, 1]


		for r in radii:
			r_new, z_new = update(r)
			rs = np.concatenate((rs, r_new))
			zs = np.concatenate((zs, z_new))

		return rs, zs



def main():
	surface = Surface("../initFiles/axes/defaultAxis.txt", 128, 32, 1.0)

	radii = np.linspace(0.0,1.2,3)

	start = time.time()

	N = 200
	coil_data, coil_params = CoilSet.get_initial_data(surface, input_file="../../tests/validateWithFocus/validate_focus.hdf5")
	rs, zs = Poincare.getPoincarePoints(N, 0.0, radii, surface, False, coil_data, coil_params)

	end = time.time()
	print(end - start)

	plt.plot(rs,zs,'ko', markersize=1)
	plt.show()

if __name__ == "__main__":
	main()
