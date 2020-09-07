import jax.numpy as np
import numpy as npo
from jax.experimental.ode import odeint
from functools import partial
from jax.config import config
import jax
from jax import jit
import matplotlib.pyplot as plt
import sys
import time
import tables as tb
sys.path.append("..")

from surface.Surface import Surface
from coils.CoilSet import CoilSet
from lossFunctions.DefaultLoss import default_loss, lhd_saddle_B
config.update("jax_enable_x64", True)


PI = np.pi

class Poincare():

	@jit
	def computeB(I, dl, l, r, zeta, z):
		"""
			Inputs:

			r, zeta, z : The coordinates of the point we want the magnetic field at. Cylindrical coordinates.

			Outputs: 

			B_z, B_zeta, B_z : the magnetic field components at the input coordinates created by the currents in the coils. Cylindrical coordinates.
		"""
		x = r * np.cos(zeta)
		y = r * np.sin(zeta)
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
		B_r = B_x * np.cos(zeta) + B_y * np.sin(zeta)
		B_zeta = - B_x * np.sin(zeta) + B_y * np.cos(zeta)
		return B_r, B_zeta, B_z

	def step(B_func, arr, zeta):
		r = arr[0]
		z = arr[1]
		B_r, B_zeta, B_z = B_func(r, zeta, z) 
		Fr = (B_r * r / B_zeta)
		Fz = (B_z * r / B_zeta)
		return np.asarray((Fr, Fz))


	def getPoincarePoints(N_poincare, zeta, is_frenet, coil_data, coil_params):
		"""

			NOTE: THIS ONLY WORKS FOR zeta = 0 RIGHT NOW

			Loop over N_poincare points in the plot, equally spaced in toroidal
			radius starting from the magnetic axis at a given zeta, and 
			creating a list of points where the plot punctures.

			Inputs:

			N_poincare : number of times we puncture the poincare plot, starting at radius r
			zeta : Which zeta will our poincare plot track? We are getting a poloidal cross-section at zeta.
			radii : list of radii r which we start at, away from our magnetic axis. We expect this to range from 0 
			to slightly higher than 1. 

			Outputs:

			rs, zs : lists of floating point numbers where the field lines intersect at a given zeta. 

		"""

		#rs_init = [5.6,  5.6,    5.45,  5.46,  5.6, 5.6,  5.48,  5.49]
		#zs_init = [0.89, -0.89, 0.73, -0.71, 0.92, -0.92, 0.73, -0.71]
		rs_init = [5.46,  5.47]
		zs_init = [0.73, -0.71]

		rs = np.asarray([])
		zs = np.asarray([])
		I, dl, r, _ = CoilSet.get_outputs(coil_data, coil_params)
		B_func = partial(Poincare.computeB, I, dl, r)
		step_partial = jit(partial(Poincare.step, B_func))
		t_eval = np.linspace(0, 2 * PI * N_poincare, N_poincare + 1)

		@jit
		def update(r,z):
			y = np.asarray((r, z))
			sol = odeint(step_partial, y, t_eval)
			return sol[:, 0], sol[:, 1]


		for i in range(len(rs_init)):
			r = rs_init[i]
			z = zs_init[i]
			r_new, z_new = update(r, z)
			print(r_new)
			rs = np.concatenate((rs, r_new))
			zs = np.concatenate((zs, z_new))

		return rs, zs

def main():


	N = 60

	def get_all_coil_data(filename):
		with tb.open_file(filename, "r") as f:
			coil_data = f.root.metadata[0]
			fc = np.asarray(f.root.coilSeries[:, :, :])
			fr = np.asarray(f.root.rotationSeries[:, :]) # NEEDS TO BE EDITED
			params = (fc, fr)
		return coil_data, params

	coil_data_fb, coil_params_fb = get_all_coil_data("../../tests/w7x/scanold2/w7x_l3.hdf5") # 0.06
	coil_data_fil, coil_params_fil = get_all_coil_data("../../tests/w7x/scanold2/w7x_l0.hdf5")
	coil_data_fb2, coil_params_fb2 = get_all_coil_data("../../tests/w7x/scanold2/w7x_l7.hdf5") # 0.12

	"""
	rs_fb, zs_fb = Poincare.getPoincarePoints(N, 0.0, False, coil_data_fb, coil_params_fb) 
	npo.save("rs_w7x_fb_islands.npy", npo.asarray(rs_fb))
	npo.save("zs_w7x_fb_islands.npy", npo.asarray(zs_fb))

	rs_fil, zs_fil = Poincare.getPoincarePoints(N, 0.0, False, coil_data_fb, coil_params_fil) 
	npo.save("rs_w7x_fil_islands.npy", npo.asarray(rs_fil))
	npo.save("zs_w7x_fil_islands.npy", npo.asarray(zs_fil))
	"""
	
	
	rs_fil2, zs_fil2 = Poincare.getPoincarePoints(N, 0.0, False, coil_data_fb2, coil_params_fil) 
	npo.save("rs_w7x_fil2_islands2.npy", npo.asarray(rs_fil2))
	npo.save("zs_w7x_fil2_islands2.npy", npo.asarray(zs_fil2))

if __name__ == "__main__":
	main()
