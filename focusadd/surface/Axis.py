import numpy as np
import math as m
from jax.config import config

config.update("jax_enable_x64", True)

PI = m.pi


class Axis:

    """ Represents the stellarator magnetic axis. """

    def __init__(self, read_axis_data, N_zeta, res=20):
        """ 
		
		Initializes axis from Fourier series, calculates real-space coordinates.

		The 

		Parameters:
		xc (np.array(float)): x-cosine coefficients of axis, size N_F+1
		xs (np.array(float)): x-sine coefficients of axis, size N_F+1
		yc (np.array(float)): y-cosine coefficients of axis, size N_F+1
		ys (np.array(float)): y-sine coefficients of axis, size N_F+1
		zc (np.array(float)): z-cosine coefficients of axis, size N_F+1
		zs (np.array(float)): z-sine coefficients of axis, size N_F+1
		N_zeta (int): Number of axis gridpoints in the toroidal direction, actually is N_zeta+1

		"""
        xc, xs, yc, ys, zc, zs, epsilon, minor_rad, N_rotate, zeta_off = read_axis_data
        self.xc = xc
        self.xs = xs
        self.yc = yc
        self.ys = ys
        self.zc = zc
        self.zs = zs
        self.NF = len(self.xc) - 1
        self.N_zeta = N_zeta
        self.NZR = self.N_zeta * res
        self.res = res
        self.zeta = np.linspace(0, 2 * PI, self.NZR + 1)
        self.epsilon = epsilon
        self.a = minor_rad
        self.NR = N_rotate
        self.zeta_off = zeta_off
        self.init_axis()

    def init_axis(self):
        """
		
		Performs the calculations required to initialize the Axis. This includes
		calculating the real-space values, the frenet frame, the length of the axis, 
		the torsion, and the derivatives of the frenet vectors. 

		"""

        self.compute_xyz()
        self.compute_x1y1z1()
        self.compute_x2y2z2()
        self.compute_x3y3z3()
        self.compute_dsdz()
        self.compute_frenet()
        self.compute_torsion()
        self.compute_mean_torsion()
        self.compute_curvature()
        self.compute_dBdz()
        self.compute_dNdz()
        self.calc_alpha()
        self.calc_frame()

    def compute_xyz(self):
        """ From the Fourier harmonics of the axis, computes the real-space coordinates of the axis. """
        x = np.zeros(self.NZR + 1)
        y = np.zeros(self.NZR + 1)
        z = np.zeros(self.NZR + 1)
        for m in range(self.NF + 1):
            arg = m * self.zeta
            x += self.xc[m] * np.cos(arg) + self.xs[m] * np.sin(arg)
            y += self.yc[m] * np.cos(arg) + self.ys[m] * np.sin(arg)
            z += self.zc[m] * np.cos(arg) + self.zs[m] * np.sin(arg)
        self.x = x
        self.y = y
        self.z = z
        self.r = np.concatenate(
            (self.x[:, np.newaxis], self.y[:, np.newaxis], self.z[:, np.newaxis]),
            axis=1,
        )

    def get_xyz(self):
        """ Returns the real-space coordinates of the axis """
        return self.x[:: self.res], self.y[:: self.res], self.z[:: self.res]

    def get_r(self):
        """ Returns the real-space coordinates of the axis in a single vector """
        return self.r[:: self.res, :]

    def get_r_from_zeta(self, zeta):
        """ Computes the real-space position of the axis for a single zeta. """
        x = 0.0
        y = 0.0
        z = 0.0
        for m in range(self.NF + 1):
            arg = m * zeta
            x += self.xc[m] * np.cos(arg) + self.xs[m] * np.sin(arg)
            y += self.yc[m] * np.cos(arg) + self.ys[m] * np.sin(arg)
            z += self.zc[m] * np.cos(arg) + self.zs[m] * np.sin(arg)
        return x, y, z

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
        """ Computes the first derivative of the real-space position with respect to zeta """
        x1 = np.zeros(self.NZR + 1)
        y1 = np.zeros(self.NZR + 1)
        z1 = np.zeros(self.NZR + 1)
        for m in range(self.NF + 1):
            arg = m * self.zeta
            x1 += -m * self.xc[m] * np.sin(arg) + m * self.xs[m] * np.cos(arg)
            y1 += -m * self.yc[m] * np.sin(arg) + m * self.ys[m] * np.cos(arg)
            z1 += -m * self.zc[m] * np.sin(arg) + m * self.zs[m] * np.cos(arg)
        self.x1 = x1
        self.y1 = y1
        self.z1 = z1
        self.r1 = np.concatenate(
            (self.x1[:, np.newaxis], self.y1[:, np.newaxis], self.z1[:, np.newaxis]),
            axis=1,
        )

    def get_r1(self):
        """ Returns the first derivative of the real-space position in a single N_zeta+1 by 3 vector """
        return self.r1[:: self.res, :]

    def compute_x2y2z2(self):
        """ Computes the second derivative of the real-space position with respect to zeta """
        x2 = np.zeros(self.NZR + 1)
        y2 = np.zeros(self.NZR + 1)
        z2 = np.zeros(self.NZR + 1)
        for m in range(self.NF + 1):
            arg = m * self.zeta
            x2 += -(m ** 2) * self.xc[m] * np.cos(arg) - (m ** 2) * self.xs[m] * np.sin(
                arg
            )
            y2 += -(m ** 2) * self.yc[m] * np.cos(arg) - (m ** 2) * self.ys[m] * np.sin(
                arg
            )
            z2 += -(m ** 2) * self.zc[m] * np.cos(arg) - (m ** 2) * self.zs[m] * np.sin(
                arg
            )
        self.x2 = x2
        self.y2 = y2
        self.z2 = z2
        self.r2 = np.concatenate(
            (self.x2[:, np.newaxis], self.y2[:, np.newaxis], self.z2[:, np.newaxis]),
            axis=1,
        )

    def get_r2(self):
        """ Returns the second derivative of the real-space position in a single N_zeta+1 by 3 vector """
        return self.r2[:: self.res, :]

    def compute_x3y3z3(self):
        """ Computes the third derivative of the real-space position with respect to zeta """
        x3 = np.zeros(self.NZR + 1)
        y3 = np.zeros(self.NZR + 1)
        z3 = np.zeros(self.NZR + 1)
        for m in range(self.NF + 1):
            arg = m * self.zeta
            x3 += (m ** 3) * self.xc[m] * np.sin(arg) - (m ** 3) * self.xs[m] * np.cos(
                arg
            )
            y3 += (m ** 3) * self.yc[m] * np.sin(arg) - (m ** 3) * self.ys[m] * np.cos(
                arg
            )
            z3 += (m ** 3) * self.zc[m] * np.sin(arg) - (m ** 3) * self.zs[m] * np.cos(
                arg
            )
        self.x3 = x3
        self.y3 = y3
        self.z3 = z3
        self.r3 = np.concatenate(
            (self.x3[:, np.newaxis], self.y3[:, np.newaxis], self.z3[:, np.newaxis]),
            axis=1,
        )

    def get_r3(self):
        """ Returns the third derivative of the real-space position in a single N_zeta+1 by 3 vector """
        return self.r3[:: self.res, :]

    def get_zeta(self):
        """ 
		zeta is a length N_zeta+1 vector which is equally spaced between 0 and 2pi.
		The first and last elements represent the same position on the axis.
		"""
        return self.zeta[:: self.res]

    def compute_dsdz(self):
        """
		Computes |dr/d_zeta|
		"""
        self.dsdz = np.linalg.norm(self.r1, axis=1)

    def get_dsdz(self):
        """
		Returns |dr/d_zeta|
		"""
        return self.dsdz[:: self.res, :]

    def compute_tangent(self):
        """ 
		Computes the tangent vector of the axis. Uses the equation 

		T = dr/d_zeta / |dr / d_zeta|

		"""
        a0 = self.dsdz
        self.tangent = self.r1 / a0[:, np.newaxis]

    def get_tangent(self):
        """ Returns the tangent vector of the axis """
        return self.tangent[:: self.res, :]

    def compute_normal(self):
        """ 
		Computes the normal vector of the axis. Uses the equation

		N = dT/ds / |dT/ds| 
		where 
		dT/ds = (r_2 - T (T . r2)) / r1**2
		which gives 
		N = r_2 - T(T . r2) / |r_2 - T(T . r2)|

		"""
        a1 = (
            self.x2 * self.tangent[:, 0]
            + self.y2 * self.tangent[:, 1]
            + self.z2 * self.tangent[:, 2]
        )
        N = self.r2 - self.tangent * a1[:, np.newaxis]
        norm = np.linalg.norm(N, axis=1)
        self.normal = N / norm[:, np.newaxis]

    def get_normal(self):
        """ Returns the normal vector of the axis """
        return self.normal[:: self.res, :]

    def compute_binormal(self):
        """ 

		Computes the binormal vector of the axis

		B = T x N

		"""
        self.binormal = np.cross(self.tangent, self.normal)

    def get_binormal(self):
        """ Returns the binormal vector of the axis """
        return self.binormal[:: self.res, :]

    def compute_torsion(self):
        """
		Computes the torsion of the axis using the formula
		
		tau = ( r1 x r2 ) . r3 / |r1 x r2|**2

		"""
        r1 = self.r1
        r2 = self.r2
        r3 = self.r3
        cross12 = np.cross(r1, r2)
        top = (
            cross12[:, 0] * r3[:, 0]
            + cross12[:, 1] * r3[:, 1]
            + cross12[:, 2] * r3[:, 2]
        )
        bottom = np.linalg.norm(cross12, axis=1) ** 2
        self.torsion = top / bottom

    def get_torsion(self):
        """ Returns the torsion of the axis as a function of zeta """

        return self.torsion[:: self.res]

    def compute_mean_torsion(self):
        """ 
		Computes the mean torsion value of the axis.

		Because the axis is periodic, the last element is not used in computing the torsion.

		"""
        self.mean_torsion = np.mean(self.torsion[:-1])

    def get_mean_torsion(self):
        """ Returns the mean torsion of the axis """
        return self.mean_torsion

    def compute_curvature(self):
        """ 

		Computes the curvature of the axis. Uses the formula

		kappa = | r1 x r2 | / |r1|**3

		"""

        r1 = self.r1
        r2 = self.r2
        cross12 = np.cross(r1, r2)
        top = np.linalg.norm(cross12, axis=1)
        bottom = np.linalg.norm(r1, axis=1) ** 3
        self.curvature = top / bottom

    def get_curvature(self):
        return self.curvature[:: self.res]

    def compute_dNdz(self):
        """
		
		Computes the derivative dN/dzeta using the Frenet-Serret equations.

		This equation is

		dN/dz = (- kappa * T + tau * B) * ds/dz

		"""

        self.dNdz = (
            -self.curvature[:, np.newaxis] * self.tangent
            + self.torsion[:, np.newaxis] * self.binormal
        ) * self.dsdz[:, np.newaxis]

    def get_dNdz(self):
        return self.dNdz[:: self.res, :]

    def compute_dBdz(self):
        """
		
		Computes the derivative dB/dzeta using the Frenet-Serret equations.

		This equation is

		dB/dz = - tau * N * ds/dz

		"""
        self.dBdz = (
            -self.torsion[:, np.newaxis] * self.normal * self.dsdz[:, np.newaxis]
        )

    def get_dBdz(self):
        return self.dBdz[:: self.res, :]

    def calc_alpha(self):
        """
		Alpha is the angle by which we rotate the Normal and Binormal vectors of 
		the axis to get the orientation of the elliptical surface we are using. We will
		ultimately rotate N and B by a rotation matrix with angle alpha.

		The angle alpha is defined by the equation

		d alpha / d zeta = N_{rotate} / 2 - (tau - <tau>)

		where tau is the torsion of the axis at a given zeta and <tau> is the averaged torsion.
		Now we see why we needed to compute the average axis torsion <tau> in Axis! 

		To compute alpha, we integrate alpha over zeta. 

		Subtracting off <tau> will enforce that alpha is closed in 2pi.

		The factor of 2 in N_rotate is due to elliptical symmetry in the surface cross-section.
		"""
        tau = self.torsion
        av_tau = self.mean_torsion
        d_zeta = 2.0 * PI / self.NZR
        tau = tau - av_tau  # subtracts off <tau>
        torsionInt = (
            np.cumsum(tau) - tau
        ) * d_zeta  # if i didn't subtract off tau this would be from
        zeta = self.zeta
        self.alpha = 0.5 * self.NR * zeta + self.zeta_off + torsionInt

    def get_alpha(self):
        """ Returns the angle alpha by which the ellipse frame is rotated relative to the normal and binormal """
        return self.alpha[:: self.res]

    def calc_frame(self):
        """ 
		Calculates the vectors v1 and v2 which are the ellipse frame. The normal and 
		binormal get rotated by alpha. 
		"""
        alpha = self.alpha
        calpha = np.cos(alpha)
        salpha = np.sin(alpha)
        N = self.normal
        B = self.binormal
        self.v1 = calpha[:, np.newaxis] * N - salpha[:, np.newaxis] * B
        self.v2 = salpha[:, np.newaxis] * N + calpha[:, np.newaxis] * B

    def get_frame(self):
        """ Returns the vectors v1 and v2 which give the ellipse frame for a given zeta. """
        return self.v1[:: self.res, :], self.v2[:: self.res, :]
