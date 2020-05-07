import jax.numpy as np
from jax import jit
import numpy as numpy
from jax.ops import index, index_add, index_update
import math as m
import tables as tb
from functools import partial


PI = m.pi

class SaddleCoilSet:

	"""
	SaddleCoilSet is a class which represents the toroidal and saddle coils surrounding a surface. 

	The toroidal coils are ultimately represented by a set of positions in space
	which do not change. 

	The saddle coils are ultimately represented by a Fourier series in zeta and theta. These are 
	closed curves in zeta and theta which lie on a (zeta, theta) surface surrounding the actual 
	plasma surface.

	The (zeta, theta) surface is described by a Fourier transform which maps (zeta, theta) -> (x,y,z)


	"""

	def __init__(self,surface,input_file=None,args_dict=None):
		"""
		There are two methods of initializing the coil set. Both require a surface which the coils surround. 

		The first method of initialization involves an HDF5 file which stores the coil data and metadata. We can supply
		input_file and this tells us a path to the coil data. 

		The second method of initialization involves reading the coils in from args_dict, which is a dictionary
		of the coil metadata. From this metadata and the surface we can initialize the coils around the surface. 
		"""

		if input_file is not None:
			with tb.open_file(input_file,'r') as f:
				# # toroidal coils, # saddle coils, # toroidal segments, # saddle segments,
				# winding-surface number of fourier components in zeta,
				# winding-surface number of fourier components in theta
				self.NC_t, self.NC_s, self.NS_t, self.NS_s, self.NF_s, self.WSNFZ, self.WSNFT, self.rc_t, self.rc_s = f.root.metadata[0]
				# NC_t x NS_t + 1 x 3 -> position of each toroidal coil
				self.r_t = np.asarray(f.root.r_t[:,:,:])
				# 2 x 2 x NC_s x NF_s + 1 -> each saddle coil has a fourier series in zeta and theta (2) and sin and cos (2)
				self.f_s = np.asarray(f.root.f_s[:,:,:,:]) 
				# NC_s -> each saddle coil has a different current
				self.I_s = np.asarray(f.root.I_s[:]) 
				# 3 x 2 x 2 x WSNFZ + 1 x WSNFT + 1 -> the mapping from (zeta, theta) -> (x,y,z)
				# (3) for xyz, (2) for sin(m zeta) or cos(m zeta), (2) for sin(n theta) or cos(n theta)
				self.winding_surface_fourier = np.asarray(f.root.winding_fourier[:,:,:,:,:])
		elif args_dict is not None:
			self.NC_t = args_dict['numCoilsToroidal']
			NC_s_zeta = args_dict['numCoilsSaddleZeta']
			NC_s_theta = args_dict['numCoilsSaddleTheta']
			self.NC_s = NC_s_zeta * NC_s_theta
			self.NS_t = args_dict['numSegmentsToroidal']
			self.NS_s = args_dict['numSegmentsSaddle']
			self.NF_s = args_dict['numFourierSaddle']
			self.rc_t = args_dict['radiusToroidal']
			self.rc_s = args_dict['initialradiusSaddle']
			self.r_ws = args_dict['radius_winding_surface']
			self.WSNFZ = args_dict['windingSurfaceNumberFourierZeta']
			self.WSNFT = args_dict['windingSurfaceNumberFourierTheta']
			self.r_t = surface.calc_r_coils(self.NC_t,self.NS_t,self.rc_t)
			self.f_s = self.initialize_fourier_saddle_coils(NC_s_zeta,NC_s_theta)
			# initialize the saddle coils to have a current of 1
			self.I_s = np.ones(self.NC_s)
			# compute the fourier decomposition of the surface
			r_surface = surface.calc_r_coils(256,128,self.r_ws)[:,0:-1,:] # using calc_r_coils as a proxy for the surface
			self.winding_surface_fourier = self.compute_surface_fourier_series(r_surface)
		else:
			raise Exception("No file or args_dict passed to initialize coil set.")
		self.r_t_middle = (self.r_t[:,1:,:] + self.r_t[:,:-1,:]) / 2.
		self.dl_t = self.r_t[:,1:,:] - self.r_t[:,:-1,:]
		self.I_t = np.ones(self.NC_t)
		self.set_params((self.f_s,self.I_s))

	def initialize_fourier_saddle_coils(self,NC_s_zeta,NC_s_theta):
		result = np.zeros((2, 2, self.NC_s, self.NF_s + 1))
		# update the zeta + theta [:], cosine [1], all coils[:], zeroth fourier [0] to be laid out on a grid
		# then add an initial circular shape to zeta + theta [:], sin and cosine [:], all coils [:], and 1st fourier [1]
		theta = np.linspace(0, 2*np.pi, NC_s_theta + 1)[0:-1]
		zeta = np.linspace(0, 2*np.pi, NC_s_zeta + 1)[0:-1]
		zeta, theta = np.meshgrid(zeta,theta)
		result = index_update(result,index[0,1,:,0], zeta.reshape(-1))
		result = index_update(result,index[1,1,:,0], theta.reshape(-1))
		result = index_update(result,index[:,:,:,1], self.rc_s)
		return result


	def compute_surface_fourier_series(self,r_surface):
		"""
		Inputs: r_surface is a NZ x NT x 3 array which has x,y,z as a function of zeta and theta.

		Outputs a 3 x 2 x 2 x WSNFZ + 1 x WSNFT + 1 array which contains the Fourier components of the surface.

		"""
		NZ = r_surface.shape[0]
		NT = r_surface.shape[1]
		x_s = r_surface[:,0:-1,0]
		y_s = r_surface[:,0:-1,1]
		z_s = r_surface[:,0:-1,2]

		result = np.zeros((3, 2, 2, self.WSNFZ + 1, self.WSNFT + 1))

		theta = np.linspace(0,2*PI,NT+1)[0:NT]
		zeta = np.linspace(0,2*PI,NZ+1)[0:NZ]

		# X^{cc}_{0,0} terms for x,y,z
		result = index_update(result,index[:,1,1,0,0], np.mean(r_surface,axis=(0,1)))
		
		for m in range(1, self.WSNFZ + 1):
			# X_{cc}_{m,0}
			result = index_update(result, index[:,1,1,m,0], 2.0 * \
				np.mean(r_surface[:,:,:] * \
					np.cos(m * zeta)[:,np.newaxis,np.newaxis], axis=(0,1)))
			# X_{sc}_{m,0}
			result = index_update(result, index[:,0,1,m,0], 2.0 * \
				np.mean(r_surface[:,:,:] * \
					np.sin(m * zeta)[:,np.newaxis,np.newaxis], axis=(0,1)))

		for n in range(1, self.WSNFT + 1):
			# X_{cc}_{0,n}
			result = index_update(result, index[:,1,1,0,n], 2.0 * \
				np.mean(r_surface[:,:,:] * \
					np.cos(n * theta)[np.newaxis,:,np.newaxis], axis=(0,1)))
			# X_{cs}_{0,n}
			result = index_update(result, index[:,1,0,0,n], 2.0 * \
				np.mean(r_surface[:,:,:] * \
					np.sin(n * theta)[np.newaxis,:,np.newaxis], axis=(0,1)))

		for m in range(1, self.WSNFZ + 1):
			for n in range(1,self.WSNFT + 1):
				# X_{ss}_{m,n}
				result = index_update(result, index[:,0,0,m,n], 4.0 * \
					np.mean(r_surface[:,:,:] * \
						np.sin(n * theta)[np.newaxis,:,np.newaxis] * \
						np.sin(m * zeta)[:,np.newaxis,np.newaxis], axis=(0,1)))
				# X_{cs}_{m,n}
				result = index_update(result, index[:,1,0,m,n], 4.0 * \
					np.mean(r_surface[:,:,:] * \
						np.sin(n * theta)[np.newaxis,:,np.newaxis] * \
						np.cos(m * zeta)[:,np.newaxis,np.newaxis], axis=(0,1)))
				# X_{sc}_{m,n}
				result = index_update(result, index[:,0,1,m,n], 4.0 * \
					np.mean(r_surface[:,:,:] * \
						np.cos(n * theta)[np.newaxis,:,np.newaxis] * \
						np.sin(m * zeta)[:,np.newaxis,np.newaxis], axis=(0,1)))
				# X_{cc}_{m,n}
				result = index_update(result, index[:,1,1,m,n], 4.0 * \
					np.mean(r_surface[:,:,:] * \
						np.cos(n * theta)[np.newaxis,:,np.newaxis] * \
						np.cos(m * zeta)[:,np.newaxis,np.newaxis], axis=(0,1)))
		return result

	def write(self, output_file):
		""" Write coils in HDF5 output format.
		Input:

		output_file (string): Path to outputfile, string should include .hdf5 format


		"""
		with tb.open_file(output_file, 'w') as f:
			metadata = numpy.dtype([('NC_t',int),('NC_s',int),('NS_t',int),('NS_s',int),('NF_s',int),('WSNFZ',int),('WSNFT',int),('rc_t',float),('rc_s',float)])
			arr = numpy.array([(self.NC_t, self.NC_s, self.NS_t, self.NS_s, self.NF_s, self.WSNFZ, self.WSNFT, self.rc_t, self.rc_s)],dtype=metadata)
			f.create_table('/','metadata',metadata)
			f.root.metadata.append(arr)
			f.create_array('/','r_t',numpy.asarray(self.r_t))
			f.create_array('/','f_s',numpy.asarray(self.f_s))
			f.create_array('/','I_s',numpy.asarray(self.I_s))
			f.create_array('/','winding_fourier',numpy.asarray(self.winding_surface_fourier))

	def get_params(self):
		return (self.f_s,self.I_s)


	def set_params(self, params):
		self.f_s, self.I_s = params
		# do stuff to compute self.r_s and self.r_s_middle and length
		self.r_s = self.compute_r_s(self.f_s,self.winding_surface_fourier) # NC_s x NS_s + 1 x 3
		self.r_s_middle = (self.r_s[:,1:,:] + self.r_s[:,:-1,:]) / 2.
		self.dl_s = self.r_s[:,1:,:] - self.r_s[:,:-1,:]
		self.saddle_length = self.compute_saddle_length(self.dl_s)

	def compute_angles(self,f_s):
		angle = np.linspace(0,2*PI,self.NS_s+1)
		zeta = np.zeros((self.NC_s,self.NS_s+1))
		theta = np.zeros((self.NC_s,self.NS_s+1))
		for m in range(self.NF_s+1):
			arg = m * angle
			carg = np.cos(arg)
			sarg = np.sin(arg)
			zeta += f_s[0,1,:,np.newaxis,m] * carg[np.newaxis,:] + f_s[0,0,:,np.newaxis,m] * sarg[np.newaxis,:]
			theta += f_s[1,1,:,np.newaxis,m] * carg[np.newaxis,:] + f_s[1,0,:,np.newaxis,m] * sarg[np.newaxis,:]
		return zeta, theta

	def compute_r_s(self,f_s,f_ws):
		theta_s, zeta_s = self.compute_angles(f_s)
		r_s = self.map_angles_to_xyz(zeta_s, theta_s, f_ws, self.NC_s,self.NS_s)
		return r_s

	def map_angles_to_xyz(self,zeta_s, theta_s, f_ws, NC, NS):
		xyz = np.zeros((NC, NS+1,3))
		for m in range(self.WSNFZ+1):
			for n in range(self.WSNFT+1):
				arg_m = m * zeta_s
				arg_n = n * theta_s
				sin_m = np.sin(arg_m)
				cos_m = np.cos(arg_m)
				sin_n = np.sin(arg_n)
				cos_n = np.cos(arg_n)
				xyz += f_ws[np.newaxis,np.newaxis,:,1,1,m,n] * cos_m[:,:,np.newaxis] * cos_n[:,:,np.newaxis] + \
					f_ws[np.newaxis,np.newaxis,:,1,0,m,n] * cos_m[:,:,np.newaxis] * sin_n[:,:,np.newaxis] + \
					f_ws[np.newaxis,np.newaxis,:,0,1,m,n] * sin_m[:,:,np.newaxis] * cos_n[:,:,np.newaxis] + \
					f_ws[np.newaxis,np.newaxis,:,0,0,m,n] * sin_m[:,:,np.newaxis] * sin_n[:,:,np.newaxis]
		return xyz

	def compute_saddle_length(self,dl_s):
		return np.sum(np.linalg.norm(dl_s,axis=-1))

	def get_saddle_length(self):
		return self.saddle_length

	def get_r_t(self):
		return self.r_t

	def get_r_t_middle(self):
		return self.r_t_middle

	def get_r_s(self):
		return self.r_s

	def get_r_s_middle(self):
		return self.r_s_middle

	def get_dl_t(self):
		return self.dl_t

	def get_dl_s(self):
		return self.dl_s

	def get_I_t(self):
		return self.I_t

	def get_I_s(self):
		return self.I_s

	def get_winding_surface_fourier(self):
		return self.winding_surface_fourier



