import argparse
import time
import sys
sys.path.append("../..")
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
#from jax.config import config
#config.update("jax_enable_x64",True)


PI = math.pi
def setArgs():
	parser = argparse.ArgumentParser()
	parser.add_argument("-n","--numIter", help="Number of iterations by the optimizer", default=1000)
	parser.add_argument("-nt","--numTheta", help="Number of gridpoints in theta (poloidal angle) on the magnetic surface", default=64)
	parser.add_argument("-nz","--numZeta", help="Number of gridpoints in zeta (toroidal angle) on the magnetic surface", default=256)
	parser.add_argument("-nc","--numCoils", help="Number of coils", default=32)
	parser.add_argument("-ns","--numSegments", help="Number of segments in each coil", default=128)
	parser.add_argument("-nfc","--numFourierCoils", help="Number of Fourier Components describing each coil", default=5)
	parser.add_argument("-nnr","--numNormalRotate", help="Number of filaments in the (rotated) normal direction for each multi-build coil", default=1)
	parser.add_argument("-nbr","--numBinormalRotate", help="Number of filaments in the (rotated) binormal direction for each multi-build coil", default=1)
	parser.add_argument("-nfr","--numFourierRotate", help="Number of Fourier Components describing the rotation relative to the torsion vector of each coil", default=0)
	parser.add_argument("-ln","--lengthNormal", help="Length between each coil in the (rotated) normal direction", default=0.00)
	parser.add_argument("-lb","--lengthBinormal", help="Length between each coil in the (rotated) binormal direction", default=0.00)
	parser.add_argument("-rc","--radiusCoil", help="Radius of coils", default=2.0)
	parser.add_argument("-rs","--radiusSurface", help="Radius of surface", default=1.0)
	parser.add_argument("-nr","--numRotate", help="Number of rotations of each finite-build coil", default=0)
	parser.add_argument("-lr","--learningRate", help="Learning Rate of SGD, ODEFlow, Newtons Method", default=0.0001)
	parser.add_argument("-o" ,"--outputFile", help="Name of output file for coils", default="filamentData")
	parser.add_argument("-i" ,"--inputFile", help="Name of input file for coils", default=None)
	parser.add_argument("-w" ,"--weightLength", help="Length of weight paid to coils", default=0.1)
	parser.add_argument("-a","--axis",help="Name of axis file", default="ellipticalAxis5Rotate")
	return parser.parse_args()

def setArgsFiniteBuild():
	parser = argparse.ArgumentParser()
	parser.add_argument("-n","--numIter", help="Number of iterations by the optimizer", default=1000)
	parser.add_argument("-nt","--numTheta", help="Number of gridpoints in theta (poloidal angle) on the magnetic surface", default=64)
	parser.add_argument("-nz","--numZeta", help="Number of gridpoints in zeta (toroidal angle) on the magnetic surface", default=256)
	parser.add_argument("-nc","--numCoils", help="Number of coils", default=32)
	parser.add_argument("-ns","--numSegments", help="Number of segments in each coil", default=128)
	parser.add_argument("-nfc","--numFourierCoils", help="Number of Fourier Components describing each coil", default=5)
	parser.add_argument("-nnr","--numNormalRotate", help="Number of filaments in the (rotated) normal direction for each multi-build coil", default=3)
	parser.add_argument("-nbr","--numBinormalRotate", help="Number of filaments in the (rotated) binormal direction for each multi-build coil", default=3)
	parser.add_argument("-nfr","--numFourierRotate", help="Number of Fourier Components describing the rotation relative to the torsion vector of each coil", default=0)
	parser.add_argument("-ln","--lengthNormal", help="Length between each coil in the (rotated) normal direction", default=0.015)
	parser.add_argument("-lb","--lengthBinormal", help="Length between each coil in the (rotated) binormal direction", default=0.015)
	parser.add_argument("-rc","--radiusCoil", help="Radius of coils", default=2.0)
	parser.add_argument("-rs","--radiusSurface", help="Radius of surface", default=1.0)
	parser.add_argument("-nr","--numRotate", help="Number of rotations of each finite-build coil", default=0)
	parser.add_argument("-lr","--learningRate", help="Learning Rate of SGD, ODEFlow, Newtons Method", default=0.0001)
	parser.add_argument("-o" ,"--outputFile", help="Name of output file for coils", default="filamentData")
	parser.add_argument("-i" ,"--inputFile", help="Name of input file for coils", default=None)
	parser.add_argument("-w" ,"--weightLength", help="Length of weight paid to coils", default=0.1)
	parser.add_argument("-a","--axis",help="Name of axis file", default="ellipticalAxis5Rotate")
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

def endRun(output_file,loss_vals,loss_vals_finite_build,coilSet,params):
	with open("{}.txt".format(output_file), 'w') as f:
		wr = csv.writer(f, quoting=csv.QUOTE_ALL)
		wr.writerow(loss_vals)
	with open("{}_finite_build.txt".format(output_file),'w') as f:
		wr = csv.writer(f, quoting=csv.QUOTE_ALL)
		wr.writerow(loss_vals_finite_build)
	coilSet.set_params(params)
	coilSet.write("{}.hdf5".format(output_file))


def main():
	# Initialize the arguments to be used by the program
	args = setArgs()
	args_finite_build = setArgsFiniteBuild()

	# Create the surface
	surface = Surface("../../initFiles/axes/{}.txt".format(args.axis), int(args.numZeta), int(args.numTheta), float(args.radiusSurface))

	args_dict = create_args_dict(args)
	args_dict_finite_build = create_args_dict(args_finite_build)

	input_file = args.inputFile
	#output_file = 'coils/saved/{}.hdf5'.format(args.outputFile)
	output_file = args.outputFile


	if input_file is not None:
		coilSet = CoilSet(surface,input_file='{}.hdf5'.format(input_file))
		coilSet_finite_build = CoilSet(surface,input_file='{}_finite_build.hdf5'.format(input_file))
	else:
		coilSet = CoilSet(surface,args_dict = args_dict)
		coilSet_finite_build = CoilSet(surface,args_dict = args_dict_finite_build)

	l = DefaultLoss(surface, coilSet, weight_length=float(args.weightLength))
	l_finite_build = DefaultLoss(surface, coilSet_finite_build, weight_length=float(args.weightLength))
	optim = GD(l, learning_rate=float(args.learningRate))

	params = coilSet.get_params()
	param0_finite_build, param1_finite_build = coilSet_finite_build.get_params()
	zeroRotateParam = np.zeros(param1_finite_build.shape)

	loss_vals = []
	loss_vals_finite_build = []
	start = time.time()
	try:
		# PERFORM OPTIMIZATION
		for i in range(int(args.numIter)):
			print(i)
			# what would loss value be if using non-filamentary coils
			param0, _ = params
			loss_vals_finite_build.append(l_finite_build.loss((param0,zeroRotateParam)))
			
			# loss_val is for old params, params is new params
			loss_val, params = optim.step(params) 
			loss_vals.append(loss_val)
			print(loss_val)
	except KeyboardInterrupt as e:
		endRun(output_file, loss_vals, loss_vals_finite_build,coilSet,params)
		raise e	
	end = time.time()
	print(end - start)
	
	endRun(output_file,loss_vals,loss_vals_finite_build,coilSet,params)











if __name__ == "__main__":
	main()
