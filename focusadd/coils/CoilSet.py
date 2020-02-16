import jax.numpy as np
from jax.ops import index, index_add
import math as m

PI = m.pi

class CoilSet:

	def __init__(self,surface,input_file=None,args_dict=None):
		if input_file is not None:
			pass
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
			self.rc = args_dict['radiusCoil']
			self.NR = args_dict['numRotate']
			self.r_central = surface.calc_r_coils(self.NC,self.NS,self.rc) # Position of central coil
			self.fc = self.compute_coil_fourierSeries(self.r_central)
			self.fr = np.zeros((self.NC,self.NFR))
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

	def write_coils(self, output_file):
		""" Write coils in HDF5 output format"""
		pass

	def calc_length(self):
		pass

	def compute_coil_fourierSeries(self,r_central):
		x = r_central[:,:,0] # NC x NS
		y = r_central[:,:,1]
		z = r_central[:,:,2]
		xc = np.zeros(self.NC,self.NF)
		yc = np.zeros(self.NC,self.NF)
		zc = np.zeros(self.NC,self.NF)
		xs = np.zeros(self.NC,self.NF)
		ys = np.zeros(self.NC,self.NF)
		zs = np.zeros(self.NC,self.NF)
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
		return [xc,yc,zc,xs,yz,zs]






	def compute_frenet(self):
		""" 
		Computes the tangent, normal, and binormal of the axis.

		These functions assume you compute the tangent first, then the normal, 
		then the binormal. 

		"""
		self.compute_tangent()
		self.compute_normal()
		self.compute_binormal()

	def compute_x1y1z1(self):
		x1 = np.zeros(self.N_zeta+1)
		y1 = np.zeros(self.N_zeta+1)
		z1 = np.zeros(self.N_zeta+1)
		for m in range(len(self.xc)):
			arg = m * self.zeta
			x1 += -m * self.xc[m] * np.sin(arg) + m * self.xs[m] * np.cos(arg)
			y1 += -m * self.yc[m] * np.sin(arg) + m * self.ys[m] * np.cos(arg)
			z1 += -m * self.zc[m] * np.sin(arg) + m * self.zs[m] * np.cos(arg)
		self.x1 = x1
		self.y1 = y1 
		self.z1 = z1

	def compute_x2y2z2(self):
		x2 = np.zeros(self.N_zeta+1)
		y2 = np.zeros(self.N_zeta+1)
		z2 = np.zeros(self.N_zeta+1)
		for m in range(len(self.xc)):
			arg = m * self.zeta
			x2 += -m**2 * self.xc[m] * np.cos(arg) - m**2 * self.xs[m] * np.sin(arg)
			y2 += -m**2 * self.yc[m] * np.cos(arg) - m**2 * self.ys[m] * np.sin(arg)
			z2 += -m**2 * self.zc[m] * np.cos(arg) - m**2 * self.zs[m] * np.sin(arg)
		self.x2 = x2
		self.y2 = y2 
		self.z2 = z2

	def compute_x3y3z3(self):
		x3 = np.zeros(self.N_zeta+1)
		y3 = np.zeros(self.N_zeta+1)
		z3 = np.zeros(self.N_zeta+1)
		for m in range(len(self.xc)):
			arg = m * self.zeta
			x3 += m**3 * self.xc[m] * np.sin(arg) - m**3 * self.xs[m] * np.cos(arg)
			y3 += m**3 * self.yc[m] * np.sin(arg) - m**3 * self.ys[m] * np.cos(arg)
			z3 += m**3 * self.zc[m] * np.sin(arg) - m**3 * self.zs[m] * np.cos(arg)
		self.x3 = x3
		self.y3 = y3 
		self.z3 = z

	def compute_torsion(self):
		r1 = self.get_r1()
		r2 = self.get_r2()
		r3 = self.get_r3()
		cross12 = np.cross(r1, r2)
		top = cross12[:,0] * r3[:,0] + cross12[:,1] * r3[:,1] + cross12[:,2] * r3[:,2]
		bottom = np.linalg.norm(cross12,axis=1)
		self.torsion = top / bottom

	def compute_integrated_torsion(self):
		pass

	def compute_curvature(self):
		r1 = self.get_r1()
		r2 = self.get_r2()
		cross12 = np.cross(r1, r2)
		top = np.linalg.norm(cross12, axis=1)
		bottom = np.linalg.norm(r1,axis=1)
		self.curvature = top / bottom

	def compute_integrated_curvature(self):
		pass

	def compute_dsdz(self):
		x1, y1, z1 = self.x1, self.y1, self.z1
		self.dsdz = np.sqrt(x1**2 + y1**2 + z1**2) # magnitude of first derivative of curve

	def compute_dNdz(self):
		self.dNdz = (- self.curvature[:,np.newaxis] * self.tangent + self.torsion[:,np.newaxis] * self.binormal ) * self.dsdz[:,np.newaxis]

	def compute_dBdz(self):
		self.dBdz = - self.torsion[:,np.newaxis] * self.normal * self.dsdz[:,np.newaxis]

	def compute_tangent(self):
		a0 = self.get_dsdz()
		top = np.concatenate((self.x1[:,np.newaxis],self.y1[:,np.newaxis],self.z1[:,np.newaxis]),axis=1)
		self.tangent = top / a0[:,np.newaxis]

	def compute_normal(self):
		a1 = self.x2 * self.tangent[:,0] + self.y2 * self.tangent[:,1] + self.z2 * self.tangent[:,2]  
		N = np.concatenate((self.x2[:,np.newaxis],self.y2[:,np.newaxis],self.z2[:,np.newaxis]),axis=1) - self.tangent * a1[:,np.newaxis]
		norm = np.linalg.norm(N,axis=1)
		self.normal = N / norm[:,np.newaxis]

	def compute_binormal(self):
		self.binormal = np.cross(self.tangent, self.normal)