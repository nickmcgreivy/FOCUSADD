import jax.numpy as np
from .Axis import Axis
from jax.config import config

config.update("jax_enable_x64", True)


def read_axis(filename):
    """
	Reads the magnetic axis from a file.

	Expects the filename to be in a specified form, which is the same as the default
	axis file given. 

	Parameters: 
		filename (string): A path to the file which has the axis data
		N_zeta_axis (int): The toroidal (zeta) resolution of the magnetic axis in real space
		epsilon: The ellipticity of the axis
		minor_rad: The minor radius of the axis, a
		N_rotate: Number of rotations of the axis
		zeta_off: The offset of the rotation of the surface in the ellipse relative to the zero starting point. 

	Returns: 
		axis (Axis): An axis object for the specified parameters.
	"""
    with open(filename, "r") as file:
        file.readline()
        _, _ = map(int, file.readline().split(" "))
        file.readline()
        xc = np.asarray([float(c) for c in file.readline().split(" ")])
        file.readline()
        xs = np.asarray([float(c) for c in file.readline().split(" ")])
        file.readline()
        yc = np.asarray([float(c) for c in file.readline().split(" ")])
        file.readline()
        ys = np.asarray([float(c) for c in file.readline().split(" ")])
        file.readline()
        zc = np.asarray([float(c) for c in file.readline().split(" ")])
        file.readline()
        zs = np.asarray([float(c) for c in file.readline().split(" ")])
        file.readline()
        file.readline()
        file.readline()
        file.readline()
        file.readline()
        file.readline()
        file.readline()
        epsilon, minor_rad, N_rotate, zeta_off = map(float, file.readline().split(" "))

    return xc, xs, yc, ys, zc, zs, epsilon, minor_rad, N_rotate, zeta_off


def main():
    filename = "../initFiles/axes/defaultAxis.txt"
    read_axis(filename, 64)


if __name__ == "__main__":
    main()
