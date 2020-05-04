import jax.numpy as np
from jax import jit
import numpy as numpy
from jax.ops import index, index_add
import math as m
import tables as tb
from functools import partial

PI = m.pi

class CoilSet:

	"""
	CoilSet is a class which represents all of the coils surrounding a plasma surface. The coils
	are represented by two fourier series, one for the coil winding pack centroid and one for the 
	rotation of the coils. 
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
				self.NC, self.NS, self.NF, self.NFR, self.ln, self.lb, self.NNR, self.NBR, self.rc, self.NR = f.root.metadata[0]
				self.fc = np.asarray(f.root.coilSeries[:,:,:])
				self.fr = np.asarray(f.root.rotationSeries[:,:,:])
		elif args_dict is not None:
			# INITIALIZE COILS TO DEFAULT VALUES
			self.NC = args_dict['numCoils']
			self.NS = args_dict['numSegments']
			self.NF = args_dict['numFourierCoils']
			self.NFR = args_dict['numFourierRotate']
			self.ln = args_dict['lengthNormal']
			self.lb = args_dict['lengthBinormal']
			self.NNR = args_dict['numNormalRotate']
			self.NBR = args_dict['numBinormalRotate']
			self.rc = args_dict['radiusCoil']
			self.NR = args_dict['numRotate']
			r_central = surface.calc_r_coils(self.NC,self.NS,self.rc) # Position of centroid
			self.fc = self.compute_coil_fourierSeries(r_central)
			self.fr = np.zeros((2,self.NC,self.NFR))
		else:
			raise Exception("No file or args_dict passed to initialize coil set. ")
		self.theta = np.linspace(0,2*PI,self.NS+1)
		self.I = np.ones(self.NC) / (self.NNR * self.NBR)
		self.set_params((self.fc, self.fr))

	def compute_coil_fourierSeries(self,r_central):
		""" 
		Takes a  and gives the coefficients
		of the coil fourier series in a single array 

		Inputs:
		r_central (nparray): vector of length NC x NS + 1 x 3, initial coil centroid

		Returns:
		6 x NC x NF array with the Fourier Coefficients of the initial coils
		"""
		x = r_central[:,0:-1,0] # NC x NS
		y = r_central[:,0:-1,1]
		z = r_central[:,0:-1,2]
		xc = np.zeros((self.NC,self.NF))
		yc = np.zeros((self.NC,self.NF))
		zc = np.zeros((self.NC,self.NF))
		xs = np.zeros((self.NC,self.NF))
		ys = np.zeros((self.NC,self.NF))
		zs = np.zeros((self.NC,self.NF))
		xc = index_add(xc,index[:,0],np.sum(x,axis=1) / self.NS)
		yc = index_add(yc,index[:,0],np.sum(y,axis=1) / self.NS)
		zc = index_add(zc,index[:,0],np.sum(z,axis=1) / self.NS)
		theta = np.linspace(0,2*PI,self.NS+1)[0:self.NS]
		for m in range(1,self.NF):
			xc = index_add(xc,index[:,m], 2.0 * np.sum(x * np.cos(m * theta), axis=1) / self.NS )
			yc = index_add(yc,index[:,m], 2.0 * np.sum(y * np.cos(m * theta), axis=1) / self.NS )
			zc = index_add(zc,index[:,m], 2.0 * np.sum(z * np.cos(m * theta), axis=1) / self.NS )
			xs = index_add(xs,index[:,m], 2.0 * np.sum(x * np.sin(m * theta), axis=1) / self.NS ) 
			ys = index_add(ys,index[:,m], 2.0 * np.sum(y * np.sin(m * theta), axis=1) / self.NS ) 
			zs = index_add(zs,index[:,m], 2.0 * np.sum(z * np.sin(m * theta), axis=1) / self.NS ) 
		return np.asarray([xc,yc,zc,xs,ys,zs]) # 6 x NC x NF

	def write(self, output_file):
		""" Write coils in HDF5 output format.
		Input:

		output_file (string): Path to outputfile, string should include .hdf5 format


		"""
		with tb.open_file(output_file, 'w') as f:
			metadata = numpy.dtype([('NC', int), ('NS', int),('NF', int),('NFR',int),('ln',float),('lb',float),('NNR',int),('NBR',int),('rc',float),('NR',int)])
			arr = numpy.array([(self.NC, self.NS, self.NF, self.NFR, self.ln, self.lb, self.NNR, self.NBR, self.rc, self.NR)],dtype=metadata)
			f.create_table('/','metadata',metadata)
			f.root.metadata.append(arr)
			f.create_array('/','coilSeries',numpy.asarray(self.fc))
			f.create_array('/','rotationSeries',numpy.asarray(self.fr))

	def get_params(self):
		""" Returns a tuple of the coil parameters, fourier series and rotation series"""
		return (self.fc, self.fr)

	#@partial(jit, static_argnums=(0,))
	def set_params(self, params):
		""" 
		Takes a tuple of coil parameters and sets the parameters. When the 
		parameters are reset, we need to update the other variables like the coil position, frenet frame, etc. 

		Inputs: 
		params (tuple of numpy arrays): Tuple of parameters, first array is 6 x NC x NF

		"""
		# UNPACK PARAMS
		self.fc, self.fr = params
		# COMPUTE COIL VARIABLES
		self.compute_r_central()
		self.compute_x1y1z1()
		self.compute_x2y2z2()
		self.compute_x3y3z3()
		self.compute_torsion()
		self.compute_mean_torsion()
		self.compute_dsdt()
		self.compute_length()
		self.compute_total_length()
		self.compute_frenet()
		self.compute_r() # finite build position

	def unpack_fourier(self,fc):
		"""
		Takes coil fourier series fc and unpacks it into 6 components
		"""
		xc = fc[0]
		yc = fc[1]
		zc = fc[2]
		xs = fc[3]
		ys = fc[4]
		zs = fc[5]
		return xc, yc, zc, xs, ys, zs


	def compute_r_central(self):
		""" Computes the position of the winding pack centroid using the coil fourier series """
		xc, yc, zc, xs, ys, zs = self.unpack_fourier(self.fc)
		x = np.zeros((self.NC,self.NS+1))
		y = np.zeros((self.NC,self.NS+1))
		z = np.zeros((self.NC,self.NS+1))
		for m in range(self.NF):
			arg = m * self.theta
			carg = np.cos(arg)
			sarg = np.sin(arg)
			x += xc[:,np.newaxis,m] * carg[np.newaxis,:] + xs[:,np.newaxis,m] * sarg[np.newaxis,:]
			y += yc[:,np.newaxis,m] * carg[np.newaxis,:] + ys[:,np.newaxis,m] * sarg[np.newaxis,:]
			z += zc[:,np.newaxis,m] * carg[np.newaxis,:] + zs[:,np.newaxis,m] * sarg[np.newaxis,:]
		self.r_central = np.concatenate((x[:,:,np.newaxis],y[:,:,np.newaxis],z[:,:,np.newaxis]),axis=2)

	def get_r_central(self):
		""" Returns the position of the winding pack centroid """
		return self.r_central

	def compute_x1y1z1(self):
		""" Computes a first derivative of the centroid """
		xc, yc, zc, xs, ys, zs = self.unpack_fourier(self.fc)
		x1 = np.zeros((self.NC,self.NS+1))
		y1 = np.zeros((self.NC,self.NS+1))
		z1 = np.zeros((self.NC,self.NS+1))
		for m in range(self.NF):
			arg = m * self.theta
			carg = np.cos(arg)
			sarg = np.sin(arg)
			x1 += - m * xc[:,np.newaxis,m] * sarg[np.newaxis,:] + m * xs[:,np.newaxis,m] * carg[np.newaxis,:]
			y1 += - m * yc[:,np.newaxis,m] * sarg[np.newaxis,:] + m * ys[:,np.newaxis,m] * carg[np.newaxis,:]
			z1 += - m * zc[:,np.newaxis,m] * sarg[np.newaxis,:] + m * zs[:,np.newaxis,m] * carg[np.newaxis,:]
		self.r1 = np.concatenate((x1[:,:,np.newaxis],y1[:,:,np.newaxis],z1[:,:,np.newaxis]),axis=2)

	def compute_x2y2z2(self):
		""" Computes a second derivative of the centroid """
		xc, yc, zc, xs, ys, zs = self.unpack_fourier(self.fc)
		x2 = np.zeros((self.NC,self.NS+1))
		y2 = np.zeros((self.NC,self.NS+1))
		z2 = np.zeros((self.NC,self.NS+1))
		for m in range(self.NF):
			m2 = m**2
			arg = m * self.theta
			carg = np.cos(arg)
			sarg = np.sin(arg)
			x2 += - m2 * xc[:,np.newaxis,m] * carg[np.newaxis,:] - m2 * xs[:,np.newaxis,m] * sarg[np.newaxis,:]
			y2 += - m2 * yc[:,np.newaxis,m] * carg[np.newaxis,:] - m2 * ys[:,np.newaxis,m] * sarg[np.newaxis,:]
			z2 += - m2 * zc[:,np.newaxis,m] * carg[np.newaxis,:] - m2 * zs[:,np.newaxis,m] * sarg[np.newaxis,:]
		self.r2 = np.concatenate((x2[:,:,np.newaxis],y2[:,:,np.newaxis],z2[:,:,np.newaxis]),axis=2)

	def compute_x3y3z3(self):
		""" Computes a third derivative of the centroid """
		xc, yc, zc, xs, ys, zs = self.unpack_fourier(self.fc)
		x3 = np.zeros((self.NC,self.NS+1))
		y3 = np.zeros((self.NC,self.NS+1))
		z3 = np.zeros((self.NC,self.NS+1))
		for m in range(self.NF):
			m3 = m**3
			arg = m * self.theta
			carg = np.cos(arg)
			sarg = np.sin(arg)
			x3 += m3 * xc[:,np.newaxis,m] * sarg[np.newaxis,:] - m3 * xs[:,np.newaxis,m] * carg[np.newaxis,:]
			y3 += m3 * yc[:,np.newaxis,m] * sarg[np.newaxis,:] - m3 * ys[:,np.newaxis,m] * carg[np.newaxis,:]
			z3 += m3 * zc[:,np.newaxis,m] * sarg[np.newaxis,:] - m3 * zs[:,np.newaxis,m] * carg[np.newaxis,:]
		self.r3 = np.concatenate((x3[:,:,np.newaxis],y3[:,:,np.newaxis],z3[:,:,np.newaxis]),axis=2)

	def compute_dsdt(self):
		""" Computes |dr/dtheta| """
		self.dsdt = np.linalg.norm(self.r1,axis=-1)

	def get_dsdt(self):
		""" Returns |dr/theta| """
		return self.dsdt

	def compute_length(self):
		""" Computes the length of the each coil """
		dt = 2. * PI / self.NS
		integrand = dt * self.dsdt[:,:-1] # doesn't sum over the endpoints twice
		self.length = np.sum(integrand,axis=1)

	def get_length(self):
		""" Returns the length of each coil, a NC length array """
		return self.length

	def compute_total_length(self):
		""" Computes the total length of the coil """
		self.total_length = np.sum(self.length)

	def get_total_length(self):
		""" Returns a scalar which is the total length of the NC coils """
		return self.total_length

	def get_r1(self):
		return self.r1

	def get_r2(self):
		return self.r2

	def get_r3(self):
		return self.r3

	def compute_frenet(self):
		""" Computes T, N, and B """
		self.compute_tangent()
		self.compute_normal()
		self.compute_binormal()

	def compute_tangent(self):
		""" 
		Computes the tangent vector of the coils. Uses the equation 

		T = dr/d_theta / |dr / d_theta|

		"""
		self.tangent = self.r1 / self.dsdt[:,:,np.newaxis]

	def get_tangent(self):
		return self.tangent

	def compute_normal(self):
		""" 
		Computes the normal vector of the coils. Uses the equation

		N = dT/ds / |dT/ds| 
		where 
		dT/ds = (r_2 - T (T . r2)) / r1**2
		which gives 
		N = r_2 - T(T . r2) / |r_2 - T(T . r2)|

		"""
		x2 = self.r2[:,:,0]
		y2 = self.r2[:,:,1]
		z2 = self.r2[:,:,2]
		a1 = x2 * self.tangent[:,:,0] + y2 * self.tangent[:,:,1] + z2 * self.tangent[:,:,2]  
		N = self.r2 - self.tangent * a1[:,:,np.newaxis]
		norm = np.linalg.norm(N,axis=2)
		self.normal = N / norm[:,:,np.newaxis]

	def get_normal(self):
		return self.normal

	def compute_binormal(self):
		""" 

		Computes the binormal vector of the coils

		B = T x N

		"""
		self.binormal = np.cross(self.tangent, self.normal)

	def get_binormal(self):
		return self.binormal

	def compute_frame(self):
		"""
		Computes the vectors v1 and v2 for each coil. v1 and v2 are rotated relative to
		the normal and binormal frame by an amount alpha. Alpha is parametrized by a Fourier series.
		"""
		alpha = np.zeros((self.NC, self.NS+1))
		alpha += self.theta * self.NR / 2
		#torsion = self.get_torsion()
		#mean_torsion = self.get_mean_torsion()
		#d_theta = 2. * PI / self.NS
		#torsion = torsion - mean_torsion[:,np.newaxis]
		#torsionInt = (np.cumsum(torsion,axis=-1) - torsion) * d_theta
		#alpha -= torsionInt
		Ac = self.fr[0]
		As = self.fr[1]
		for m in range(self.NFR):
			arg = self.theta * m
			carg = np.cos(arg)
			sarg = np.sin(arg)
			alpha += Ac[:,np.newaxis,m] * carg[np.newaxis,:] + As[:,np.newaxis,m] * sarg[np.newaxis,:]
		calpha = np.cos(alpha)
		salpha = np.sin(alpha)
		self.v1 = calpha[:,:,np.newaxis] * self.normal - salpha[:,:,np.newaxis] * self.binormal
		self.v2 = salpha[:,:,np.newaxis] * self.normal + calpha[:,:,np.newaxis] * self.binormal

	def get_frame(self):
		return self.v1, self.v2

	def compute_r(self):
		"""
		Computes the position of the multi-filament coils.

		self.r is a NC x NS + 1 x NNR x NBR x 3 array which holds the coil endpoints
		self.dl is a NC x NS x NNR x NBR x 3 array which computes the length of the NS segments
		self.r_middle is a NC x NS x NNR x NBR x 3 array which computes the midpoint of each of the NS segments

		"""
		self.compute_frame()
		r = np.zeros((self.NC,self.NS+1,self.NNR,self.NBR,3))
		r += self.r_central[:,:,np.newaxis,np.newaxis,:]
		for n in range(self.NNR):
			for b in range(self.NBR):
				r = index_add(r,index[:,:,n,b,:], (n - .5*(self.NNR-1)) * self.ln * self.v1 + (b - .5*(self.NBR-1)) * self.lb * self.v2)
		self.r = r
		self.dl = (self.r[:,1:,:,:,:] - self.r[:,:-1,:,:,:])
		self.r_middle = (self.r[:,1:,:,:,:] + self.r[:,:-1,:,:,:]) / 2.

	def get_r(self):
		return self.r
	def get_dl(self):
		return self.dl
	def get_r_middle(self):
		""" self.r_middle is a NC x NS x NNR x NBR x 3 array which computes the midpoint of each of the NS segments """
		return self.r_middle
	def get_I(self):
		return self.I
	
	def compute_torsion(self):
		r1 = self.get_r1() # NC x NS+1 x 3
		r2 = self.get_r2() # NC x NS+1 x 3
		r3 = self.get_r3() # NC x NS+1 x 3
		cross12 = np.cross(r1, r2)
		top = cross12[:,:,0] * r3[:,:,0] + cross12[:,:,1] * r3[:,:,1] + cross12[:,:,2] * r3[:,:,2]
		bottom = np.linalg.norm(cross12,axis=-1)**2
		self.torsion = top / bottom # NC x NS+1

	def get_torsion(self):
		return self.torsion

	def compute_mean_torsion(self):
		self.mean_torsion = np.mean(self.torsion[:,:-1],axis=-1)

	def get_mean_torsion(self):
		return self.mean_torsion
	"""
	def compute_curvature(self):
		pass

	def compute_integrated_curvature(self):
		pass

	def compute_dNdz(self):
		pass

	def compute_dBdz(self):
		pass
	"""

	def writeXYZ(self,params,filename):
		self.set_params(params)
		with open(filename,'w') as f:
			f.write("periods {}\n".format(0))
			f.write("begin filament\n")
			f.write("FOCUSADD Coils\n")
			for i in range(self.NC):
				for n in range(self.NNR):
					for b in range(self.NBR):
						for s in range(self.NS):
							f.write("{} {} {} {}\n".format(self.r[i,s,n,b,0],self.r[i,s,n,b,1],self.r[i,s,n,b,2],self.I[i]))
						f.write("{} {} {} {} {} {}\n".format(self.r[i,self.NS,n,b,0],self.r[i,self.NS,n,b,1],self.r[i,self.NS,n,b,2],0.0,"{}{}{}".format(i,n,b),"coil/filament1/filament2"))



