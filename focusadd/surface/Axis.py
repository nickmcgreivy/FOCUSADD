

class Axis:

	""" Represents the stellarator magnetic axis. """

	def __init__(self, xc, xs, yc, ys, zc, zs, epsilon, minor_rad, N_rotate, zeta_off, N_zeta):
		""" Initializes axis from Fourier series, calculates real-space coordinates. """
		self.xc = xc
		self.xs = xs
		self.yc = yc
		self.ys = yz
		self.zc = zc
		self.zs = zs
		self.epsilon = epsilon
		self.minor_rad = minor_rad
		self.N_rotate = N_rotate
		self.zeta_off = zeta_off
		self.N_zeta = N_zeta


	def computeXYZ(xc,xs,yc,ys,zc,zs, N_zeta):
		x = np.zeros(N_zeta)
		y = np.zeros(N_zeta)
		z = np.zeros(N_zeta)
