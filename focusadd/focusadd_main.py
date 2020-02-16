import argparse
from surface.readAxis import readAxis
from surface.Surface import Surface
from surface.Axis import Axis
from surface.CoilSet import CoilSet

def setArgs():
	parser = argparse.ArgumentParser()
	parser.add_argument("-n","--numIter", help="Number of iterations by the optimizer", default=10)
	parser.add_argument("-nt","--numTheta", help="Number of gridpoints in theta (poloidal angle) on the magnetic surface", default=32)
	parser.add_argument("-nz","--numZeta", help="Number of gridpoints in zeta (toroidal angle) on the magnetic surface", default=64)
	parser.add_argument("-nc","--numCoils", help="Number of coils", default=8)
	parser.add_argument("-ns","--numSegments", help="Number of segments in each coil", default=32)
	parser.add_argument("-nfc","--numFourierCoils", help="Number of Fourier Components describing each coil", default=4)
	parser.add_argument("-nnr","--numNormalRotate", help="Number of filaments in the (rotated) normal direction for each multi-build coil", default=2)
	parser.add_argument("-nbr","--numBinormalRotate", help="Number of filaments in the (rotated) binormal direction for each multi-build coil", default=2)
	parser.add_argument("-nfr","--numFourierRotate", help="Number of Fourier Components describing the rotation relative to the torsion vector of each coil", default=4)
	parser.add_argument("-ln","--lengthNormal", help="Length between each coil in the (rotated) normal direction", default=0.01)
	parser.add_argument("-lb","--lengthBinormal", help="Length between each coil in the (rotated) binormal direction", default=0.01)
	parser.add_argument("-rc","--radiusCoil", help="Radius of coils", default=2.0)
	parser.add_argument("-rs","--radiusSurface", help="Radius of surface", default=1.0)
	return parser.parse_args()


def main():
	# Initialize the arguments to be used by the program
	args = setArgs()

	# Read and return the axis
	axis, epsilon, minor_rad, N_rotate, zeta_off = readAxis("./initFiles/axes/defaultAxis.txt",int(args.numZeta))

	# Create the surface
	surface = Surface(axis, int(args.numZeta), int(args.numTheta), epsilon, minor_rad, N_rotate, zeta_off,float(args.radiusSurface))

	coilSet = CoilSet(axis,surface,args=setArgs)






if __name__ == "__main__":
	main()