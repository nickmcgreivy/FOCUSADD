import argparse
from surface.readAxis import readAxis
from surface.Surface import Surface
from surface.Axis import Axis
from coils.CoilSet import CoilSet
import jax.numpy as np
import numpy as numpy
from lossFunctions.DefaultLoss import DefaultLoss
from optimizers.GD import GD
from optimizers.ODEFlow import ODEFlow
from optimizers.Newton import Newton
from shapeGradient.ShapeGradient import ShapeGradient
import math
import csv


PI = math.pi
def setArgs():
	parser = argparse.ArgumentParser()
	parser.add_argument("-n","--numIter", help="Number of iterations by the optimizer", default=500)
	parser.add_argument("-nt","--numTheta", help="Number of gridpoints in theta (poloidal angle) on the magnetic surface", default=32)
	parser.add_argument("-nz","--numZeta", help="Number of gridpoints in zeta (toroidal angle) on the magnetic surface", default=64)
	parser.add_argument("-nc","--numCoils", help="Number of coils", default=8)
	parser.add_argument("-ns","--numSegments", help="Number of segments in each coil", default=32)
	parser.add_argument("-nfc","--numFourierCoils", help="Number of Fourier Components describing each coil", default=4)
	parser.add_argument("-nnr","--numNormalRotate", help="Number of filaments in the (rotated) normal direction for each multi-build coil", default=1)
	parser.add_argument("-nbr","--numBinormalRotate", help="Number of filaments in the (rotated) binormal direction for each multi-build coil", default=1)
	parser.add_argument("-nfr","--numFourierRotate", help="Number of Fourier Components describing the rotation relative to the torsion vector of each coil", default=0)
	parser.add_argument("-ln","--lengthNormal", help="Length between each coil in the (rotated) normal direction", default=0.01)
	parser.add_argument("-lb","--lengthBinormal", help="Length between each coil in the (rotated) binormal direction", default=0.01)
	parser.add_argument("-rc","--radiusCoil", help="Radius of coils", default=2.0)
	parser.add_argument("-rs","--radiusSurface", help="Radius of surface", default=1.0)
	parser.add_argument("-nr","--numRotate", help="Number of rotations of each finite-build coil", default=0)
	parser.add_argument("-lr","--learningRate", help="Learning Rate of SGD, ODEFlow, Newtons Method", default=0.001)
	parser.add_argument("-o" ,"--outputFile", help="Name of output file for coils", default="simpleTest")
	parser.add_argument("-i" ,"--inputFile", help="Name of input file for coils", default=None)
	parser.add_argument("-w" ,"--weightLength", help="Length of weight paid to coils", default=0.1)
	parser.add_argument("-a","--axis",help="Name of axis file", default="defaultAxis")
	return parser.parse_args()

def create_args_dict(args):
	args_dict = {}
	args_dict['numCoils'] = int(args.numCoils)
	args_dict['numSegments'] = int(args.numSegments)
	args_dict['numFourierCoils'] = int(args.numFourierCoils)
	args_dict['numFourierRotate'] = int(args.numFourierRotate)
	args_dict['lengthNormal'] = float(args.lengthNormal)
	args_dict['lengthBinormal'] = float(args.lengthBinormal)
	args_dict['numNormalRotate'] = int(args.numNormalRotate)
	args_dict['numBinormalRotate'] = int(args.numBinormalRotate)
	args_dict['radiusCoil'] = float(args.radiusCoil)
	args_dict['numRotate'] = int(args.numRotate)
	return args_dict

def main():
	# Initialize the arguments to be used by the program
	args = setArgs()

	# Create the surface
	surface = Surface("./initFiles/axes/{}.txt".format(args.axis), int(args.numZeta), int(args.numTheta), float(args.radiusSurface))

	args_dict = create_args_dict(args)

	input_file = args.inputFile
	output_file = 'coils/saved/{}.hdf5'.format(args.outputFile)


	if input_file is not None:
		coilSet = CoilSet(surface,input_file='coils/saved/{}.hdf5'.format(input_file))
	else:
		coilSet = CoilSet(surface,args_dict = args_dict)

	l = DefaultLoss(surface, coilSet, weight_length=float(args.weightLength))
	optim = GD(l, learning_rate=float(args.learningRate))

	params = coilSet.get_params()

	loss_vals = []

	# PERFORM OPTIMIZATION
	for i in range(int(args.numIter)):
		loss_val, params = optim.step(params) # loss_val is for old params, params is new params
		loss_vals.append(loss_val)

	with open("{}.txt".format(output_file), 'wb') as f:
		wr = csv.writer(f, quoting=csv.QUOTE_ALL)
		wr.writerow(loss_vals)

	
	coilSet.set_params(params)
	coilSet.write("{}.hdf5".format(output_file))

	#shapegrad = ShapeGradient(surface, coilSet)
	#g = shapegrad.coil_gradient()












if __name__ == "__main__":
	main()
