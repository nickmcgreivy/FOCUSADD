import argparse
import time
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
from functools import partial
import jax.experimental.optimizers as op
from jax import value_and_grad
#from jax.config import config
#config.update("jax_enable_x64",True)


PI = math.pi

def args_to_op(optimizer_string, lr, mom=0.9):
	return {
	"gd":  lambda lr, _: op.sgd(lr),
	"sgd": lambda lr, _: op.sgd(lr),
	"GD":  lambda lr, _: op.sgd(lr),
	"SGD": lambda lr, _: op.sgd(lr),
	"momentum": lambda lr, mom: op.momentum(lr, mom)
	}[optimizer_string](lr, mom)

def setArgs():
	parser = argparse.ArgumentParser()
	parser.add_argument("-n","--numIter", help="Number of iterations by the optimizer", default=10, type=int)
	parser.add_argument("-nt","--numTheta", help="Number of gridpoints in theta (poloidal angle) on the magnetic surface", default=32, type=int)
	parser.add_argument("-nz","--numZeta", help="Number of gridpoints in zeta (toroidal angle) on the magnetic surface", default=64, type=int)
	parser.add_argument("-nc","--numCoils", help="Number of coils", default=8, type=int)
	parser.add_argument("-ns","--numSegments", help="Number of segments in each coil", default=32, type=int)
	parser.add_argument("-nfc","--numFourierCoils", help="Number of Fourier Components describing each coil", default=4, type=int)
	parser.add_argument("-nnr","--numNormalRotate", help="Number of filaments in the (rotated) normal direction for each multi-build coil", default=1, type=int)
	parser.add_argument("-nbr","--numBinormalRotate", help="Number of filaments in the (rotated) binormal direction for each multi-build coil", default=1, type=int)
	parser.add_argument("-nfr","--numFourierRotate", help="Number of Fourier Components describing the rotation relative to the torsion vector of each coil", default=0, type=int)
	parser.add_argument("-ln","--lengthNormal", help="Length between each coil in the (rotated) normal direction", default=0.01, type=float)
	parser.add_argument("-lb","--lengthBinormal", help="Length between each coil in the (rotated) binormal direction", default=0.01, type=float)
	parser.add_argument("-rc","--radiusCoil", help="Radius of coils", default=2.0, type=float)
	parser.add_argument("-rs","--radiusSurface", help="Radius of surface", default=1.0, type=float)
	parser.add_argument("-nr","--numRotate", help="Number of rotations of each finite-build coil", default=0, type=int)
	parser.add_argument("-lr","--learningRate", help="Learning Rate of SGD, ODEFlow, Newtons Method", default=0.001, type=float)
	parser.add_argument("-o" ,"--outputFile", help="Name of output file for coils", default="simpleTest", type=str)
	parser.add_argument("-i" ,"--inputFile", help="Name of input file for coils", default=None, type=str)
	parser.add_argument("-w" ,"--weightLength", help="Length of weight paid to coils", default=0.1, type=float)
	parser.add_argument("-a","--axis",help="Name of axis file", default="defaultAxis", type=str)
	parser.add_argument("-op","--optimizer",help="Name of optimizer. Either SGD, GD (same) or Momentum", default="GD", type=str)
	parser.add_argument("-mom","--momentum_mass",help="Momentum mass parameter.", default=0.9, type=float)
	return parser.parse_args()

def create_args_dict(args):
	args_dict = {}
	args_dict['numCoils'] = args.numCoils
	args_dict['numSegments'] = args.numSegments
	args_dict['numFourierCoils'] = args.numFourierCoils
	args_dict['numFourierRotate'] = args.numFourierRotate
	args_dict['lengthNormal'] = args.lengthNormal
	args_dict['lengthBinormal'] = args.lengthBinormal
	args_dict['numNormalRotate'] = args.numNormalRotate
	args_dict['numBinormalRotate'] = args.numBinormalRotate
	args_dict['radiusCoil'] = args.radiusCoil
	args_dict['numRotate'] = args.numRotate
	return args_dict


def update(i, opt_state, get_params, opt_update, loss):
	params = get_params(opt_state)
	loss_val, gradient = value_and_grad(loss)(params)
	return opt_update(i, gradient, opt_state), loss_val


def main():
	args = setArgs()
	args_dict = create_args_dict(args)
	input_file = args.inputFile
	output_file = args.outputFile

	surface = Surface("./initFiles/axes/{}.txt".format(args.axis), args.numZeta, args.numTheta, args.radiusSurface)

	if input_file is not None:
		coilSet = CoilSet(surface,input_file='coils/saved/{}.hdf5'.format(input_file))
	else:
		coilSet = CoilSet(surface,args_dict = args_dict)

	l = DefaultLoss(surface, coilSet, weight_length=args.weightLength)

	opt_init, opt_update, get_params = args_to_op(args.optimizer, args.learningRate, args.momentum_mass)
	opt_state = opt_init(coilSet.get_params())

	loss_vals = []
	start = time.time()

	for i in range(args.numIter):
		opt_state, loss_val = update(i, opt_state, get_params, opt_update, l.loss)
		params = get_params(opt_state)
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
