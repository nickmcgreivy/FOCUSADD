import argparse
import time
from surface.readAxis import readAxis
from surface.Surface import Surface
from surface.Axis import Axis
from coils.SaddleCoilSet import SaddleCoilSet
import jax.numpy as np
import numpy as numpy
from lossFunctions.SaddleLoss import SaddleLoss
from optimizers.GD import GD
import math
import csv
from functools import partial
#from jax.config import config
#config.update("jax_enable_x64",True)


PI = math.pi
def setArgs():
	parser = argparse.ArgumentParser()
	parser.add_argument("-n","--numIter", help="Number of iterations by the optimizer", default=10)
	parser.add_argument("-nt","--numTheta", help="Number of gridpoints in theta (poloidal angle) on the magnetic surface", default=32)
	parser.add_argument("-nz","--numZeta", help="Number of gridpoints in zeta (toroidal angle) on the magnetic surface", default=64)
	parser.add_argument("-nct","--numCoilsToroidal", help="Number of toroidal coils", default=24)
	parser.add_argument("-nst","--numSegmentsToroidal", help="Number of segments in each toroidal coil", default=32)
	parser.add_argument("-nss","--numSegmentsSaddle", help="Number of segments in each saddle coil", default=16)
	parser.add_argument("-nfs","--numFourierSaddle", help="Number of Fourier Components describing each saddle coil", default=2)
	parser.add_argument("-irs","--initialradiusSaddle", help="Initial radius of saddle coils in angular coordinates", default=0.25)
	parser.add_argument("-rs","--radiusSurface", help="Radius of surface", default=1.0)
	parser.add_argument("-rws","--radiusWindingSurface", help="Radius of winding surface", default=2.5)
	parser.add_argument("-rt","--radiusToroidalCoils", help="Radius of toroidal coils", default=3.5)
	parser.add_argument("-ncsz","--numCoilsSaddleZeta", help="Number of saddle coils initially in the zeta direction", default=5)
	parser.add_argument("-ncst","--numCoilsSaddleTheta", help="Number of saddle coils initially in the theta direction", default=4)
	parser.add_argument("-wsfz","--windingSurfaceNumberFourierZeta", help="Fourier components of winding surface in zeta direction", default=5)
	parser.add_argument("-wsft","--windingSurfaceNumberFourierTheta", help="Fourier components of winding surface in theta direction", default=5)
	parser.add_argument("-lr","--learningRate", help="Learning Rate of SGD, ODEFlow, Newtons Method", default=0.001)
	parser.add_argument("-o" ,"--outputFile", help="Name of output file for coils", default="simpleTest")
	parser.add_argument("-i" ,"--inputFile", help="Name of input file for coils", default=None)
	parser.add_argument("-w" ,"--weightLengthSaddle", help="Length of weight paid to saddle coils", default=0.01)
	parser.add_argument("-a","--axis",help="Name of axis file", default="circularAxis5Rotate")
	return parser.parse_args()

def create_args_dict(args):
	args_dict = {}
	args_dict['numCoilsToroidal'] = int(args.numCoilsToroidal)
	args_dict['numCoilsSaddleZeta'] = int(args.numCoilsSaddleZeta)
	args_dict['numCoilsSaddleTheta'] = int(args.numCoilsSaddleTheta)
	args_dict['numSegmentsToroidal'] = int(args.numSegmentsToroidal)
	args_dict['numSegmentsSaddle'] = int(args.numSegmentsSaddle)
	args_dict['numFourierSaddle'] = int(args.numFourierSaddle)
	args_dict['radiusToroidal'] = float(args.radiusToroidalCoils)
	args_dict['initialradiusSaddle'] = float(args.initialradiusSaddle)
	args_dict['radius_winding_surface'] = float(args.radiusWindingSurface)
	args_dict['windingSurfaceNumberFourierZeta'] = int(args.windingSurfaceNumberFourierZeta)
	args_dict['windingSurfaceNumberFourierTheta'] = int(args.windingSurfaceNumberFourierTheta)
	return args_dict

def main():
	# Initialize the arguments to be used by the program
	args = setArgs()

	# Create the surface
	surface = Surface("./initFiles/axes/{}.txt".format(args.axis), int(args.numZeta), int(args.numTheta), float(args.radiusSurface))

	args_dict = create_args_dict(args)

	input_file = args.inputFile
	#output_file = 'coils/saved/{}.hdf5'.format(args.outputFile)
	output_file = args.outputFile


	if input_file is not None:
		coilSet = SaddleCoilSet(surface,input_file='{}.hdf5'.format(input_file))
	else:
		coilSet = SaddleCoilSet(surface,args_dict = args_dict)


	l = SaddleLoss(surface, coilSet, weight_length=float(args.weightLengthSaddle))
	optim = GD(l, learning_rate=float(args.learningRate))

	params = coilSet.get_params()

	loss_vals = []
	start = time.time()
	# PERFORM OPTIMIZATION
	for i in range(int(args.numIter)):
		loss_val, params = optim.step(params) # loss_val is for old params, params is new params
		loss_vals.append(loss_val)
		print(loss_val)
	
	end = time.time()
	print(end - start)
	with open("{}.txt".format(output_file), 'w') as f:
		wr = csv.writer(f, quoting=csv.QUOTE_ALL)
		wr.writerow(loss_vals)

	
	coilSet.set_params(params)
	coilSet.write("{}.hdf5".format(output_file))

if __name__ == "__main__":
	main()
