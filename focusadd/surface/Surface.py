import numpy as np
import math as m
from .readAxis import read_axis
from .Axis import Axis
from jax.config import config

config.update("jax_enable_x64", True)

PI = m.pi


class Surface:

    """
	Represents the outer magnetic surface of the plasma. 
	"""

    def __init__(self, filename, num_zeta, num_theta, s, res=20):
        """
		Initializes the magnetic surface. 

		Inputs:

		filename (string): the location of the axis file
		num_zeta (int): The number of gridpoints on the surface in the toroidal direction (zeta)
		num_theta (int): The number of gridpoints on the surface in the poloidal direction (theta)
		epsilon (float): The ellepticity of the surface
		minor_rad (float): The minor radius of the surface
		N_rotate (int): The number of times the surface twists in space.
		zeta_off (float): The offset of the rotation of the ellipse at zeta = 0.
		s (float): The scale factor for the surface. In FOCUSADD this is usually set to 1.
		"""
        self.filename = filename
        self.NT = num_theta
        self.NZ = num_zeta
        self.axis = Axis(read_axis(self.filename), num_zeta, res=res)
        self.epsilon = self.axis.epsilon
        self.a = self.axis.a
        self.NR = self.axis.NR
        self.zeta_off = self.axis.zeta_off

        # self.num_zeta = num_zeta
        # self.num_theta = num_theta
        self.s = s
        self.initialize_surface()

    def initialize_surface(self):
        """
		Here we call three functions which are needed to initialize the surface. 
		"""
        self.calc_r()
        self.calc_nn()

    def calc_r(self):
        """
		The surface is a 2d toroidal surface which surrounds the axis. The surface is discretized
		into NZ+1 x NT+1 gridpoints, which are periodic in zeta and theta. 

		We compute two variables:

		self.r : NZ+1 x NT+1 x 3 array with the gridpoints
		self.r_central : NZ x NT x 3 array with the position at the center of the NZ x NT tiles in the grid.

		The equation for r is 

		r = r_axis + s * a * [sqrt(epsilon) * cos(theta) * v1(zeta) + sin(theta) * v2(zeta) / sqrt(epsilon)]

		"""
        r = np.zeros((self.NZ + 1, self.NT + 1, 3))
        sa = self.s * self.a  # multiply by scale
        zeta = self.axis.get_zeta()
        theta = np.linspace(0.0, 2.0 * PI, self.NT + 1)
        ctheta = np.cos(theta)
        stheta = np.sin(theta)
        ep = self.epsilon
        v1, v2 = self.axis.get_frame()
        r += self.axis.get_r()[:, np.newaxis, :]
        r += sa * np.sqrt(ep) * v1[:, np.newaxis, :] * ctheta[np.newaxis, :, np.newaxis]
        r += sa * v2[:, np.newaxis, :] * stheta[np.newaxis, :, np.newaxis] / np.sqrt(ep)
        self.r = r
        self.r_central = self.r[:-1, :-1, :]
        """self.r_central = (
            self.r[1:, 1:, :]
            + self.r[1:, :-1, :]
            + self.r[:-1, 1:, :]
            + self.r[:-1, :-1, :]
        ) / 4.0"""

    def get_r(self):
        """ Returns the surface positions, with shape NZ+1 x NT+1 x 3 """
        return self.r

    def get_r_central(self):
        """ Returns the surface positions, with shape NZ x NT x 3 """
        return self.r_central

    def calc_r_coils(self, num_coils, num_segments, coil_radius):
        """
		This function is used in the initialization of the coil set. We initialize the coil set a constant
		distance from the magnetic surface, and this allows us to initialize these coils a certain distance away.
		Since the surface is elliptically shaped at each cross-section, the coils will be initialized with
		and elliptical shape. 

		Inputs:
		
		num_coils (int) : the number of coils
		num_segments (int) : The number of segments in each coil. Since the coils are periodic, there are 1 more
		gridpoints than there are segments. 
		coil_radius (float) : The distance between the magnetic axis and the coils. If s, the scale of the surface, is 1,
		then setting coil_radius=2 will give us coils twice the surface radius. 

		"""

        r = np.zeros((num_coils, num_segments + 1, 3))
        resAxis2 = 100  # how high resolution is the axis
        axis2 = Axis(read_axis(self.filename), num_coils * resAxis2)
        sa = coil_radius * axis2.a
        zetaCoils = axis2.get_zeta()[:-1:resAxis2]
        theta = np.linspace(0.0, 2.0 * PI, num_segments + 1)
        ctheta = np.cos(theta)
        stheta = np.sin(theta)
        ep = axis2.epsilon
        v1, v2 = axis2.get_frame()
        r += axis2.get_r()[:-1:resAxis2, np.newaxis, :]
        r += (
            sa
            * np.sqrt(ep)
            * v1[:-1:resAxis2, np.newaxis, :]
            * ctheta[np.newaxis, :, np.newaxis]
        )
        r += (
            sa
            * v2[:-1:resAxis2, np.newaxis, :]
            * stheta[np.newaxis, :, np.newaxis]
            / np.sqrt(ep)
        )
        return r

    def calc_drdt(self):
        """
		We need dr/dtheta to compute the normal vector to the surface. 

		The equation for dr/dtheta is

		dr/theta = s * a * [-sqrt(epsilon) * sin(theta) * v1(zeta) + cos(theta) * v2(zeta) / sqrt(epsilon)]
		"""

        drdt = np.zeros((self.NZ + 1, self.NT + 1, 3))
        s = 1.0
        sa = s * self.a
        zeta = self.axis.get_zeta()
        theta = np.linspace(0.0, 2.0 * PI, self.NT + 1)
        ctheta = np.cos(theta)
        stheta = np.sin(theta)
        ep = self.epsilon
        v1, v2 = self.axis.get_frame()
        drdt -= (
            sa * np.sqrt(ep) * v1[:, np.newaxis, :] * stheta[np.newaxis, :, np.newaxis]
        )
        drdt += (
            sa * v2[:, np.newaxis, :] * ctheta[np.newaxis, :, np.newaxis] / np.sqrt(ep)
        )
        self.drdt = drdt

    def get_drdt(self):
        """ Returns dr/dtheta """
        return self.drdt

    def calc_drdz(self):
        """
		We need dr/dtheta to compute the normal vector to the surface. 

		The equation for dr/dtheta is complicated and given in the FOCUSADD
		theory document. 

		"""

        drdz = np.zeros((self.NZ + 1, self.NT + 1, 3))
        s = 1.0
        sa = s * self.a
        zeta = self.axis.get_zeta()
        theta = np.linspace(0.0, 2.0 * PI, self.NT + 1)
        ctheta = np.cos(theta)
        stheta = np.sin(theta)
        ep = self.epsilon
        drdz += self.axis.get_r1()[:, np.newaxis, :]
        alpha = self.axis.get_alpha()
        calpha = np.cos(alpha)
        salpha = np.sin(alpha)
        dNdz = self.axis.get_dNdz()
        dBdz = self.axis.get_dBdz()
        tau = self.axis.get_torsion()
        av_tau = self.axis.get_mean_torsion()
        dalphadz = self.NR / 2.0 + tau - av_tau
        dv1dz = (
            calpha[:, np.newaxis] * dNdz
            - salpha[:, np.newaxis] * dBdz
            - self.axis.get_normal() * salpha[:, np.newaxis] * dalphadz[:, np.newaxis]
            - calpha[:, np.newaxis] * dalphadz[:, np.newaxis] * self.axis.get_binormal()
        )
        dv2dz = (
            salpha[:, np.newaxis] * dNdz
            + calpha[:, np.newaxis] * dBdz
            + calpha[:, np.newaxis] * dalphadz[:, np.newaxis] * self.axis.get_normal()
            - salpha[:, np.newaxis] * dalphadz[:, np.newaxis] * self.axis.get_binormal()
        )

        drdz += (
            sa
            * np.sqrt(ep)
            * dv1dz[:, np.newaxis, :]
            * ctheta[np.newaxis, :, np.newaxis]
        )
        drdz += (
            sa
            * dv2dz[:, np.newaxis, :]
            * stheta[np.newaxis, :, np.newaxis]
            / np.sqrt(ep)
        )
        self.drdz = drdz

    def get_drdz(self):
        """ Returns dr/dzeta """
        return self.drdz

    def calc_nn(self):
        """ 
		Computes the surface area of each tile and the surface unit normal vector for each tile. 

		n = dr/dtheta x dr/dzeta / |dr/dtheta x dr/dzeta|
		"""
        self.calc_drdt()
        self.calc_drdz()
        nn = np.cross(self.drdt, self.drdz)
        nn = nn[:-1, :-1, :]
        # nn = (nn[1:, 1:, :] + nn[1:, :-1, :] + nn[:-1, 1:, :] + nn[:-1, :-1, :]) / 4.0
        self.sg = np.linalg.norm(nn * 4 * PI ** 2 / (self.NT * self.NZ), axis=2)
        self.nn = nn / np.linalg.norm(nn, axis=2)[:, :, np.newaxis]

    def get_nn(self):
        """
		Returns the surface unit normal vector. There is one
		normal vector for each tile of the surface grid, so this has length NZ x NT x 3. 
		"""
        return self.nn

    def get_sg(self):
        """ Returns the surface area for each surface grid tile. This has length NZ x NT. """
        return self.sg

    def get_axis(self):
        return self.axis

    def get_data(self):
        return
