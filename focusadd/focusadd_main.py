import argparse
import time
from surface.readAxis import read_axis
from surface.Surface import Surface
from surface.Axis import Axis
from coils.CoilSet import CoilSet
import jax.numpy as np
import numpy as numpy
from lossFunctions.DefaultLoss import default_loss
import math
import csv
from functools import partial
import jax.experimental.optimizers as op
from jax import value_and_grad, jit
from surface.readAxis import read_axis
from jax.config import config

config.update("jax_enable_x64", True)


PI = math.pi


def args_to_op(optimizer_string, lr, mom=0.9, var=0.999, eps=1e-7):
    return {
        "gd": lambda lr, *unused: op.sgd(lr),
        "sgd": lambda lr, *unused: op.sgd(lr),
        "momentum": lambda lr, mom, *unused: op.momentum(lr, mom),
        "adam": lambda lr, mom, var, eps: op.adam(lr, mom, var, eps),
    }[optimizer_string.lower()](lr, mom, var, eps)


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--num_iter",
        help="Number of iterations by the optimizer",
        default=500,
        type=int,
    )
    parser.add_argument(
        "-nt",
        "--num_theta",
        help="Number of gridpoints in theta (poloidal angle) on the magnetic surface",
        default=16,
        type=int,
    )
    parser.add_argument(
        "-nz",
        "--num_zeta",
        help="Number of gridpoints in zeta (toroidal angle) on the magnetic surface",
        default=128,
        type=int,
    )
    parser.add_argument(
        "-nc", "--num_coils", help="Number of coils", default=20, type=int
    )
    parser.add_argument(
        "-ns",
        "--num_segments",
        help="Number of segments in each coil",
        default=64,
        type=int,
    )
    parser.add_argument(
        "-nfc",
        "--num_fourier_coils",
        help="Number of Fourier Components describing each coil",
        default=6,
        type=int,
    )
    parser.add_argument(
        "-nnr",
        "--num_normal_rotate",
        help="Number of filaments in the (rotated) normal direction for each multi-build coil",
        default=1,
        type=int,
    )
    parser.add_argument(
        "-nbr",
        "--num_binormal_rotate",
        help="Number of filaments in the (rotated) binormal direction for each multi-build coil",
        default=1,
        type=int,
    )
    parser.add_argument(
        "-nfr",
        "--num_fourier_rotate",
        help="Number of Fourier Components describing the rotation relative to the torsion vector of each coil",
        default=0,
        type=int,
    )
    parser.add_argument(
        "-ln",
        "--length_normal",
        help="Length between each coil in the (rotated) normal direction",
        default=0.015,
        type=float,
    )
    parser.add_argument(
        "-lb",
        "--length_binormal",
        help="Length between each coil in the (rotated) binormal direction",
        default=0.015,
        type=float,
    )
    parser.add_argument(
        "-rc", "--radius_coil", help="Radius of coils", default=2.0, type=float
    )
    parser.add_argument(
        "-rs", "--radius_surface", help="Radius of surface", default=1.0, type=float
    )
    parser.add_argument(
        "-nr",
        "--num_rotate",
        help="Number of rotations of each finite-build coil",
        default=0,
        type=int,
    )
    parser.add_argument(
        "-lr",
        "--learning_rate_fc",
        help="Learning Rate of SGD, ODEFlow, Newtons Method",
        default=0.0001,
        type=float,
    )
    parser.add_argument(
        "-lrfr",
        "--learning_rate_fr",
        help="Learning Rate of SGD, ODEFlow, Newtons Method for coil rotation",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "-o",
        "--output_file",
        help="Name of output file for coils",
        default="simpleTest",
        type=str,
    )
    parser.add_argument(
        "-i",
        "--input_file",
        help="Name of input file for coils",
        default=None,
        type=str,
    )
    parser.add_argument(
        "-w",
        "--weight_length",
        help="Length of weight paid to coils",
        default=0.5,
        type=float,
    )
    parser.add_argument(
        "-wb",
        "--weight_B",
        help="Length of weight paid to quadratic flux",
        default=1e3,
        type=float,
    )
    parser.add_argument(
        "-a",
        "--axis",
        help="Name of axis file",
        default="ellipticalAxis4Rotate",
        type=str,
    )
    parser.add_argument(
        "-op",
        "--optimizer",
        help="Name of optimizer. Either SGD, GD (same), Momentum, or Adam",
        default="momentum",
        type=str,
    )
    parser.add_argument(
        "-mom",
        "--momentum_mass",
        help="Momentum mass parameter.",
        default=0.9,
        type=float,
    )
    parser.add_argument(
        "-res",
        "--axis_resolution",
        help="Resolution of the axis, multiplies NZ.",
        default=10,
        type=int,
    )

    return parser.parse_args()


def create_args_dict(args):
    args_dict = {}
    args_dict["numCoils"] = args.num_coils
    args_dict["numSegments"] = args.num_segments
    args_dict["numFourierCoils"] = args.num_fourier_coils
    args_dict["numFourierRotate"] = args.num_fourier_rotate
    args_dict["lengthNormal"] = args.length_normal
    args_dict["lengthBinormal"] = args.length_binormal
    args_dict["numNormalRotate"] = args.num_normal_rotate
    args_dict["numBinormalRotate"] = args.num_binormal_rotate
    args_dict["radiusCoil"] = args.radius_coil
    args_dict["numRotate"] = args.num_rotate
    return args_dict

def get_coil_data(args):
    NC = args.num_coils
    NS = args.num_segments
    NFC = args.num_fourier_coils
    NFR = args.num_fourier_rotate
    ln = args.length_normal
    lb = args.length_binormal
    NNR = args.num_normal_rotate
    NBR = args.num_binormal_rotate
    rc = args.radius_coil
    NR = args.num_rotate
    return NC, NS, NFC, NFR, ln, lb, NNR, NBR, rc, NR


def get_initial_params(args):
    input_file = args.input_file
    if args.axis.lower() == "w7x":
        assert args.num_zeta == 150
        assert args.num_theta == 20
        assert args.num_coils == 50
        # need to assert that axis has right number of points
        r = np.load("surface/w7x_r_surf.npy")
        nn = np.load("surface/w7x_nn_surf.npy")
        sg = np.load("surface/w7x_sg_surf.npy")
        surface_data = (r, nn, sg)
        if input_file is None:
            fc = np.load("surface/w7x_fc.npy")
            fr = np.zeros((2, args.num_coils, args.num_fourier_rotate))
            params = (fc, fr)
            coil_data = get_coil_data(args)
        else:
            with tb.open_file(input_file, "r") as f:
                coil_data = f.root.metadata[0]
                fc = np.asarray(f.root.coilSeries[:, :, :])
                fr = np.asarray(f.root.rotationSeries[:, :, :])
                params = (fc, fr)
    else:
        filename = "./initFiles/axes/{}.txt".format(args.axis)
        surface = Surface(
            filename,
            args.num_zeta,
            args.num_theta,
            args.radius_surface,
            res=args.axis_resolution,
        )
        if input_file is not None:
            coil_data, params = CoilSet.get_initial_data(
                surface, input_file="{}.hdf5".format(input_file)
            )
        else:
            coil_data, params = CoilSet.get_initial_data(
                surface, args_dict=create_args_dict(args)
            )
            surface_data = (surface.get_r_central(), surface.get_nn(), surface.get_sg())
    return coil_data, params, surface_data


def main():
    @jit
    def update(i, opt_state_fc, opt_state_fr):
        fc = get_params_fc(opt_state_fc)
        fr = get_params_fr(opt_state_fr)
        params = fc, fr
        w_args = (args.weight_B, args.weight_length)
        loss_val, gradient = value_and_grad(
            lambda params: default_loss(surface_data, coil_output_func, w_args, params)
        )(params)
        g_fc, g_fr = gradient
        return opt_update_fc(i, g_fc, opt_state_fc), opt_update_fr(i, g_fr, opt_state_fr), loss_val

    args = set_args()
    output_file = args.output_file
    write_file = "{}.hdf5".format(output_file)

    coil_data, init_params, surface_data = get_initial_params(args)
    fc_init, fr_init = init_params

    

    coil_output_func = partial(CoilSet.get_outputs, coil_data)

    opt_init_fc, opt_update_fc, get_params_fc = args_to_op(
        args.optimizer, args.learning_rate_fc, args.momentum_mass,
    )
    opt_init_fr, opt_update_fr, get_params_fr = args_to_op(
        args.optimizer, args.learning_rate_fr, args.momentum_mass,
    )
    opt_state_fc = opt_init_fc(fc_init)
    opt_state_fr = opt_init_fr(fr_init)

    loss_vals = []
    start = time.time()

    for i in range(args.num_iter):
        opt_state_fc, opt_state_fr, loss_val = update(i, opt_state_fc, opt_state_fr)
        loss_vals.append(loss_val)
        print(loss_val)
    end = time.time()
    print(end - start)

    with open("{}.txt".format(output_file), "w") as f:
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        wr.writerow(loss_vals)
    params = (get_params_fc(opt_state_fc), get_params_fr(opt_state_fr))
    CoilSet.write(coil_data, params, write_file)


if __name__ == "__main__":
    main()
