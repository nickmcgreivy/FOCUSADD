import jax.numpy as np
import numpy as numpy
from jax.ops import index, index_add
import math as m
import tables as tb

PI = m.pi

class CoilSet:

	def __init__(self,surface,input_file=None,args_dict=None):
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
			self.rc = args_dict['radiusCoil'] # initial radius
			self.NR = args_dict['numRotate']
			r_central = surface.calc_r_coils(self.NC,self.NS,self.rc) # Position of central coil
			self.fc = self.compute_coil_fourierSeries(r_central)
			self.fr = np.zeros((2,self.NC,self.NFR))
		else:
			raise Exception("No file or args_dict passed to initialize coil set. ")
		self.theta = np.linspace(0,2*PI,self.NS+1)
		self.I = np.ones(self.NC)
		self.set_params((self.fc, self.fr))

	def compute_coil_fourierSeries(self,r_central):
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
			xs = index_add(xs,index[:,m], 2.0 * np.sum(x * np.sin(m * theta), axis=1) / self.NS ) # - 
			ys = index_add(ys,index[:,m], 2.0 * np.sum(y * np.sin(m * theta), axis=1) / self.NS ) # - 
			zs = index_add(zs,index[:,m], 2.0 * np.sum(z * np.sin(m * theta), axis=1) / self.NS ) # - 
		return np.asarray([xc,yc,zc,xs,ys,zs]) # 6 x NC x NF

	def write(self, output_file):
		""" Write coils in HDF5 output format"""
		with tb.open_file(output_file, 'w') as f:
			metadata = numpy.dtype([('NC', int), ('NS', int),('NF', int),('NFR',int),('ln',float),('lb',float),('NNR',int),('NBR',int),('rc',float),('NR',int)])
			arr = numpy.array([(self.NC, self.NS, self.NF, self.NFR, self.ln, self.lb, self.NNR, self.NBR, self.rc, self.NR)],dtype=metadata)
			f.create_table('/','metadata',metadata)
			f.root.metadata.append(arr)
			f.create_array('/','coilSeries',numpy.asarray(self.fc))
			f.create_array('/','rotationSeries',numpy.asarray(self.fr))

	def get_params(self):
		return (self.fc, self.fr)

	def set_params(self, params):
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

	def unpack_fourier(self,f):
		xc = f[0]
		yc = f[1]
		zc = f[2]
		xs = f[3]
		ys = f[4]
		zs = f[5]
		return xc, yc, zc, xs, ys, zs


	def compute_r_central(self):
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
		return self.r_central

	def compute_x1y1z1(self):
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
		self.dsdt = np.linalg.norm(self.r1,axis=-1)

	def get_dsdt(self):
		return self.dsdt

	def compute_length(self):
		dt = 2. * PI / self.NS
		integrand = dt * self.dsdt[:,0:-1]
		self.length = np.sum(integrand,axis=1)
		dl = self.r_central[:,1:] - self.r_central[:,:-1]
		dr = np.linalg.norm(dl,axis=-1)
		length2 = np.sum(dr,axis=-1)


	def get_length(self):
		return self.length

	def compute_total_length(self):
		self.total_length = np.sum(self.length)

	def get_total_length(self):
		return self.total_length

	def get_r1(self):
		return self.r1

	def get_r2(self):
		return self.r2

	def get_r3(self):
		return self.r3

	def compute_frenet(self):
		self.compute_tangent()
		self.compute_normal()
		self.compute_binormal()

	def compute_tangent(self):
		a0 = self.dsdt
		self.tangent = self.r1 / a0[:,:,np.newaxis]

	def compute_normal(self):
		x2 = self.r2[:,:,0]
		y2 = self.r2[:,:,1]
		z2 = self.r2[:,:,2]
		a1 = x2 * self.tangent[:,:,0] + y2 * self.tangent[:,:,1] + z2 * self.tangent[:,:,2]  
		N = self.r2 - self.tangent * a1[:,:,np.newaxis]
		norm = np.linalg.norm(N,axis=2)
		self.normal = N / norm[:,:,np.newaxis]

	def compute_binormal(self):
		self.binormal = np.cross(self.tangent, self.normal)

	def get_tangent(self):
		return self.tangent

	def get_normal(self):
		return self.normal

	def get_binormal(self):
		return self.binormal

	def compute_r(self):
		self.compute_frame()
		r = np.zeros((self.NC,self.NS+1,self.NNR,self.NBR,3))
		r += self.r_central[:,:,np.newaxis,np.newaxis,:]
		for n in range(self.NNR):
			for b in range(self.NBR):
				r = index_add(r,index[:,:,n,b,:], (n - .5*(self.NNR-1)) * self.ln * self.v1 + (b - .5*(self.NBR-1)) * self.lb * self.v2)
		self.r = r
		self.dl = (self.r[:,1:,:,:,:] - self.r[:,:-1,:,:,:])
		self.r_middle = (self.r[:,1:,:,:,:] + self.r[:,:-1,:,:,:]) / 2.

	def compute_frame(self):
		alpha = np.zeros((self.NC, self.NS+1))
		alpha += self.theta * self.NR / 2
		torsion = self.axis.get_torsion()
		mean_torsion = self.axis.get_mean_torsion()
		d_theta = 2. * PI / self.NS
		torsion = torsion - mean_torsion
		torsionInt = (np.cumsum(torsion,axis=-1) - torsion) * d_theta
		alpha += torsionInt
		rc = self.fr[0]
		rs = self.fr[1]
		for m in range(self.NFR):
			arg = self.theta * m / 2
			carg = np.cos(arg)
			sarg = np.sin(arg)
			alpha += rc[:,np.newaxis,m] * carg[np.newaxis,:] + rs[:,np.newaxis,m] * sarg[np.newaxis,:]
		calpha = np.cos(alpha)
		salpha = np.sin(alpha)
		self.v1 = calpha[:,:,np.newaxis] * self.normal + salpha[:,:,np.newaxis] * self.binormal
		self.v2 = -salpha[:,:,np.newaxis] * self.normal + calpha[:,:,np.newaxis] * self.binormal
		
	def get_frame(self):
		return self.v1, self.v2

	def get_r(self):
		return self.r
	def get_dl(self):
		return self.dl
	def get_r_middle(self):
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
		self.mean_torsion = np.mean(self.torsion[:,:-1])

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

