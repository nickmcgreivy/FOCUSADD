from mayavi import mlab
import jax.numpy as np
import tables as tb
import sys
sys.path.append("../../../")
from focusadd.coils.CoilSet import CoilSet
from focusadd.lossFunctions.LossFunction import LossFunction
import matplotlib.pyplot as plt
import pdb
from functools import partial
from jax import grad, vmap

def get_all_coil_data(filename):
	with tb.open_file(filename, "r") as f:
		coil_data = f.root.metadata[0]
		fc = np.asarray(f.root.coilSeries[:, :, :])
		fr = np.asarray(f.root.rotationSeries[:, :])
		params = (fc, fr)
	return coil_data, params

def filament_real_space(p, theta):
	r = np.zeros((3, p.shape[1], NS))
	for m in range(p.shape[2]):
		r += p[:3, :, None, m] * np.cos(m * theta)[None, :, :] + p[3:, :, None, m] * np.sin(m * theta)[None, :, :]
	return np.transpose(r, (1, 2, 0))


def fourier_to_real_space(p, theta):
	# p is 6 x NF, theta is scalar
	r = np.zeros(3)
	for m in range(p.shape[1]):
		r += p[:3, m] * np.cos(m * theta) + p[3:, m] * np.sin(m * theta)
	return r

def objective_scalar(fc_new, r_fil, theta_i):
	# fc_new is 6 x NF, r_fil and theta_i are scalars
	r_new = fourier_to_real_space(fc_new, theta_i)
	return np.linalg.norm(r_new - r_fil)

def find_minimum_theta_scalar(fc_new, r_fil, theta_i):
	f = partial(objective_scalar, fc_new, r_fil)
	f_prime = grad(f)
	f_primeprime = grad(f_prime)
	for n in range(n_iter):
		theta_i = theta_i - alpha * np.nan_to_num(f_prime(theta_i) / (f_primeprime(theta_i) + epsilon))
	return theta_i

find_minimum_theta_single_coil = vmap(find_minimum_theta_scalar, (None, 0, 0), 0)
find_minimum_theta_all_coils = vmap(find_minimum_theta_single_coil, (1, 0, 0), 0)

NS = 64
n_iter = 100
alpha = 0.8

_, params_fil = get_all_coil_data("../../../tests/w7x/scan2/w7x_l0.hdf5")
fc_fil, _ = params_fil
NC = fc_fil.shape[1]
r_fil = filament_real_space(fc_fil, np.tile(np.linspace(0, 2 * np.pi, NS+1)[:-1], (NC,1)))


nums =  ["0", "6",    "1",   "7",    "9",  "2",  "8",    "5",   "3"  ]
sizes = [0.0, 0.0175, 0.035, 0.0525, 0.06, 0.07, 0.0875, 0.105, 0.14 ]
eps =   [0.0, 3e4,    1e4,   2e3,    2e3,  1e3,  5e2,    4e2,   2e2  ]
epsilon = 0.0

mean_delta_rs = [0.0]
max_delta_rs = [0.0]

for i in range(1,len(nums)):
	epsilon = eps[i]
	theta_i = np.tile(np.linspace(0, 2 * np.pi, NS+1)[:-1], (NC,1))
	_, params_new = get_all_coil_data("../../../tests/w7x/scan2/w7x_l{}.hdf5".format(nums[i]))
	fc_new, _ = params_new
	theta_i = find_minimum_theta_all_coils(fc_new, r_fil, theta_i)


	print("Size is {}".format(sizes[i]))
	print("The original delta r 1 is:")
	print(np.mean(np.linalg.norm(r_fil - filament_real_space(fc_new, np.tile(np.linspace(0, 2 * np.pi, NS+1)[:-1], (NC,1))), axis=-1)))
	print("The new delta r 1 is (with minimization):")
	print(np.mean(np.linalg.norm(r_fil - filament_real_space(fc_new, theta_i), axis=-1)))
	mean_delta_rs.append(np.mean(np.linalg.norm(r_fil - filament_real_space(fc_new, theta_i), axis=-1)))
	print("The max distance is")
	print(np.max(np.linalg.norm(r_fil - filament_real_space(fc_new, theta_i), axis=-1)))
	max_delta_rs.append(np.max(np.linalg.norm(r_fil - filament_real_space(fc_new, theta_i), axis=-1)))


	difference = np.linalg.norm(r_fil - filament_real_space(fc_new, np.tile(np.linspace(0, 2 * np.pi, NS+1)[:-1], (NC,1))), axis=-1) - np.linalg.norm(r_fil - filament_real_space(fc_new, theta_i), axis=-1)
	new_diff = difference[difference > 0]
	larger = np.ravel(difference).shape[0] - new_diff.shape[0]
	if larger > 0:
		print(larger)

np.save("w7x_sizes.npy", np.asarray(sizes))
np.save("w7x_mean_delta_rs.npy", np.asarray(mean_delta_rs))
np.save("w7x_max_delta_rs.npy", np.asarray(max_delta_rs))

