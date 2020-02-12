import numpy as np

def readAxis(filename, N_zeta_axis):
	"""
	Reads the magnetic axis from a file.

	Expects the filename to be in a specified form, which is the same as the default
	axis file given. 

	Parameters: 
		filename (string): A path to the file which has the axis data
		N_zeta_axis (int): The toroidal (zeta) resolution of the magnetic axis in real space

	Returns: 
		axis (Axis): An axis object for the specified parameters.
	"""
	with open(filename, "r") as file:
		file.readline()
		NFaxis, NP = map(int,file.readline().split(" ")) 
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
		epsilon, minor_rad, N_rotate, zeta_off = map(float, file.readline().split(' '))

	return Axis()
		


			

def main():
	filename = "../initFiles/axes/defaultAxis.txt"
	readAxis(filename)

if __name__ == "__main__":
	main()
