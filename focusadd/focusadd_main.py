import argparse


def setArgs():
	parser = argparse.ArgumentParser()
	parser.add_argument("-n","--numIter", help="Number of iterations by the optimizer", default=10)
	parser.add_argument("-nt","--numTheta", help="Number of gridpoints in theta (poloidal angle) on the magnetic surface", default=32)
	parser.add_argument("-nz","--numZeta", help="Number of gridpoints in zeta (toroidal angle) on the magnetic surface", default=64)
	return parser.parse_args()


def main():
	# Initialize the arguments to be used by the program
	args = setArgs()

	# Read the axis




if __name__ == "__main__":
	main()