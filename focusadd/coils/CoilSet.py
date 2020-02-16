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
			# READ IN HDF5 FORMAT 
			# DON'T IMPLEMENT YET
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
			self.fr = np.zeros((6,self.NC,self.NFR))
		else:
			raise Exception("No file or args_dict passed to initialize coil set. ")
		#self.set_params((self.fc, self.fr))

	def get_r_central(self):
		return self.r_central

	def set_params(self, params):
		# UNPACK PARAMS
		self.fc, self.fr = params
		# COMPUTE THINGS
		self.compute_r()
		self.compute_x1y1z1()
		self.compute_x2y2z2()
		self.compute_x3y3z3()
		self.compute_dsdz()
		self.compute_frenet()
		self.compute_torsion()
		self.compute_curvature()
		self.compute_integrated_torsion()
		self.compute_integrated_curvature()

	def get_params(self):
		return (self.fc, self.fr)

	def compute_r(self):
		pass

	def get_r(self):
		return self.r

	def get_params(self):
		return self.fc, self.fr

	def write(self, output_file):
		""" Write coils in HDF5 output format"""
		with tb.open_file(output_file, 'w') as f:
			metadata = numpy.dtype([('NC', int), ('NS', int),('NF', int),('NFR',int),('ln',float),('lb',float),('NNR',int),('NBR',int),('rc',float),('NR',int)])
			arr = numpy.array([(self.NC, self.NS, self.NF, self.NFR, self.ln, self.lb, self.NNR, self.NBR, self.rc, self.NR)],dtype=metadata)
			f.create_table('/','metadata',metadata)
			f.root.metadata.append(arr)
			f.create_array('/','coilSeries',numpy.asarray(self.fc))
			f.create_array('/','rotationSeries',numpy.asarray(self.fr))



	def calc_length(self):
		pass

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
			xc = index_add(xc,index[:,m],-2.0 * np.sum(x * np.cos(m * theta), axis=1) / self.NS )
			yc = index_add(yc,index[:,m],-2.0 * np.sum(y * np.cos(m * theta), axis=1) / self.NS )
			zc = index_add(zc,index[:,m],-2.0 * np.sum(z * np.cos(m * theta), axis=1) / self.NS )
			xs = index_add(xs,index[:,m],-2.0 * np.sum(x * np.sin(m * theta), axis=1) / self.NS )
			ys = index_add(ys,index[:,m],-2.0 * np.sum(y * np.sin(m * theta), axis=1) / self.NS )
			zs = index_add(zs,index[:,m],-2.0 * np.sum(z * np.sin(m * theta), axis=1) / self.NS )
		return np.asarray([xc,yc,zc,xs,ys,zs]) # 6 x NC x NF

	def compute_frenet(self):
		self.compute_tangent()
		self.compute_normal()
		self.compute_binormal()

	def compute_x1y1z1(self):
		pass

	def compute_x2y2z2(self):
		pass

	def compute_x3y3z3(self):
		pass

	def compute_torsion(self):
		pass

	def compute_integrated_torsion(self):
		pass

	def compute_curvature(self):
		pass

	def compute_integrated_curvature(self):
		pass

	def compute_dsdz(self):
		pass

	def compute_dNdz(self):
		pass

	def compute_dBdz(self):
		pass

	def compute_tangent(self):
		pass

	def compute_normal(self):
		pass

	def compute_binormal(self):
		pass