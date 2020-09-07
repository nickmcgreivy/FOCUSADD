from mayavi import mlab
import numpy as np
import tables as tb
import sys
sys.path.append("../../../")
from focusadd.coils.CoilSet import CoilSet
from focusadd.lossFunctions.LossFunction import LossFunction
import matplotlib.pyplot as plt
import pdb

def read_w7x_data():
	r_surf = np.load("../../../focusadd/initFiles/w7x/w7x_r_surf.npy")
	nn = np.load("../../../focusadd/initFiles/w7x/w7x_nn_surf.npy")
	sg = np.load("../../../focusadd/initFiles/w7x/w7x_sg_surf.npy")
	fc_init = np.load("../../../focusadd/initFiles/w7x/w7x_fc.npy")
	return r_surf, nn, sg, fc_init


def plot_surface(r):
	x = r[:,:,0]
	y = r[:,:,1]
	z = r[:,:,2]
	p = mlab.mesh(x,y,z,color=(0.8,0.0,0.0))
	return p
	

def get_all_coil_data(filename):
	with tb.open_file(filename, "r") as f:
		coil_data = f.root.metadata[0]
		fc = np.asarray(f.root.coilSeries[:, :, :])
		fr = np.asarray(f.root.rotationSeries[:, :]) # NEEDS TO BE EDITED
		params = (fc, fr)
	return coil_data, params


def plot_coils_centroid(r_coils, color=(0.0, 0.0, 0.8)):
	r_coils = np.concatenate((r_coils[:, :, :], r_coils[:, 0:2, :]),axis=1)
	for ic in range(r_coils.shape[0]):
		p = mlab.plot3d(r_coils[ic,:,0], r_coils[ic,:,1], r_coils[ic,:,2], tube_radius=0.02, color = color)#, line_width = 0.01,)
	return p


def plot_coils(r_coils, color=(0.0, 0.0, 0.8)):
	r_coils = np.concatenate((r_coils[:, :, :, :, :], r_coils[:, 0:1, :, :, :]),axis=1)
	for ic in range(r_coils.shape[0]):
		for n in range(r_coils.shape[2]):
			for b in range(r_coils.shape[3]):
				p = mlab.plot3d(r_coils[ic,:,n,b,0], r_coils[ic,:,n,b,1], r_coils[ic,:,n,b,2], tube_radius=0.02, color = color)#, line_width = 0.01,)
	return p

coil_sizes = np.load("w7x_sizes.npy") * 300
delta_rs = np.load("w7x_mean_delta_rs.npy") * 1000
max_delta_rs = np.load("w7x_max_delta_rs.npy") * 1000

plt.loglog(coil_sizes[1:], delta_rs[1:])
plt.show()

font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 16}

plt.rc('font', **font)
plt.figure(figsize=(7,5.2))
plt.plot(coil_sizes , delta_rs, linestyle='--', color="grey", linewidth=1)
plt.scatter(coil_sizes, delta_rs, label='mean')
plt.scatter(coil_sizes, max_delta_rs, c="red", label='max')
plt.ylim([0.0,50.0])
plt.ylabel("$\Delta r$ (mm)")
plt.xlabel("Coil size (cm)")
plt.legend()
plt.savefig("w7x_delta_r.eps")
plt.show()

I = np.ones(50)

coil_data_fil, params_fil = get_all_coil_data("../../../tests/w7x/scanold2/w7x_l0.hdf5") # fil
I_fil, dl_fil, r_fil, _ = CoilSet.get_outputs(coil_data_fil, params_fil)
centroid_fil = CoilSet.get_r_centroid(coil_data_fil, params_fil)

coil_data_1, params_1 = get_all_coil_data("../../../tests/w7x/scanold2/w7x_l1.hdf5") # 0.02
I_1, dl_1, r_1, _ = CoilSet.get_outputs(coil_data_1, params_1)
centroid_1 = CoilSet.get_r_centroid(coil_data_1, params_1)

coil_data_2, params_2 = get_all_coil_data("../../../tests/w7x/scanold2/w7x_l2.hdf5") # 0.04
I_2, dl_2, r_2, _ = CoilSet.get_outputs(coil_data_2, params_2)
centroid_2 = CoilSet.get_r_centroid(coil_data_2, params_2)

coil_data_3, params_3 = get_all_coil_data("../../../tests/w7x/scanold2/w7x_l3.hdf5") # 0.06
I_3, dl_3, r_3, _ = CoilSet.get_outputs(coil_data_3, params_3)
centroid_3 = CoilSet.get_r_centroid(coil_data_3, params_3)

coil_data_4, params_4 = get_all_coil_data("../../../tests/w7x/scanold2/w7x_l4.hdf5") # 0.07
I_4, dl_4, r_4, _ = CoilSet.get_outputs(coil_data_4, params_4)
centroid_4 = CoilSet.get_r_centroid(coil_data_4, params_4)

coil_data_5, params_5 = get_all_coil_data("../../../tests/w7x/scanold2/w7x_l5.hdf5") # 0.08
I_5, dl_5, r_5, _ = CoilSet.get_outputs(coil_data_5, params_5)
centroid_5 = CoilSet.get_r_centroid(coil_data_5, params_5)

coil_data_6, params_6 = get_all_coil_data("../../../tests/w7x/scanold2/w7x_l6.hdf5") # 0.10
I_6, dl_6, r_6, _ = CoilSet.get_outputs(coil_data_6, params_6)
centroid_6 = CoilSet.get_r_centroid(coil_data_6, params_6)

coil_data_7, params_7 = get_all_coil_data("../../../tests/w7x/scanold2/w7x_l7.hdf5") # 0.12
I_7, dl_7, r_7, _ = CoilSet.get_outputs(coil_data_7, params_7)
centroid_7 = CoilSet.get_r_centroid(coil_data_7, params_7)

coil_data_8, params_8 = get_all_coil_data("../../../tests/w7x/scanold2/w7x_l8.hdf5") # 0.14
I_8, dl_8, r_8, _ = CoilSet.get_outputs(coil_data_8, params_8)
centroid_8 = CoilSet.get_r_centroid(coil_data_8, params_8)

coil_data_9, params_9 = get_all_coil_data("../../../tests/w7x/scanold2/w7x_l9.hdf5") # 0.16
I_9, dl_9, r_9, _ = CoilSet.get_outputs(coil_data_9, params_9)
centroid_9 = CoilSet.get_r_centroid(coil_data_9, params_9)




r_surf, nn, sg, _ = read_w7x_data()
mlab.options.offscreen = True
mlab.figure(size=(1600,1200), bgcolor=(1,1,1))

#p = plot_surface(r_surf)

"""
L = 0
U = 5

p = plot_coils(r_9, color=(0.8,0.0,0.0))
#mlab.view(azimuth=30.0, elevation=90, distance=6.0, focalpoint=(5.5,1.8,0.0), roll=None, reset_roll=True)

mlab.savefig('w7x_c9.eps', figure=mlab.gcf())
mlab.clf()
"""


loss_1 = LossFunction.normalized_error(r_surf, I_1, dl_1, r_1, nn, sg)
loss_2 = LossFunction.normalized_error(r_surf, I_2, dl_2, r_2, nn, sg)
loss_3 = LossFunction.normalized_error(r_surf, I_3, dl_3, r_3, nn, sg)
loss_4 = LossFunction.normalized_error(r_surf, I_4, dl_4, r_4, nn, sg)
loss_5 = LossFunction.normalized_error(r_surf, I_5, dl_5, r_5, nn, sg)
loss_6 = LossFunction.normalized_error(r_surf, I_6, dl_6, r_6, nn, sg)
loss_7 = LossFunction.normalized_error(r_surf, I_7, dl_7, r_7, nn, sg)
loss_8 = LossFunction.normalized_error(r_surf, I_8, dl_8, r_8, nn, sg)
loss_9 = LossFunction.normalized_error(r_surf, I_9, dl_9, r_9, nn, sg)
I_fil_1, dl_fil_1, r_fil_1, _ = CoilSet.get_outputs(coil_data_1, params_fil)
I_fil_2, dl_fil_2, r_fil_2, _ = CoilSet.get_outputs(coil_data_2, params_fil)
I_fil_3, dl_fil_3, r_fil_3, _ = CoilSet.get_outputs(coil_data_3, params_fil)
I_fil_4, dl_fil_4, r_fil_4, _ = CoilSet.get_outputs(coil_data_4, params_fil)
I_fil_5, dl_fil_5, r_fil_5, _ = CoilSet.get_outputs(coil_data_5, params_fil)
I_fil_6, dl_fil_6, r_fil_6, _ = CoilSet.get_outputs(coil_data_6, params_fil)
I_fil_7, dl_fil_7, r_fil_7, _ = CoilSet.get_outputs(coil_data_7, params_fil)
I_fil_8, dl_fil_8, r_fil_8, _ = CoilSet.get_outputs(coil_data_8, params_fil)
I_fil_9, dl_fil_9, r_fil_9, _ = CoilSet.get_outputs(coil_data_9, params_fil)
loss_fil_1 = LossFunction.normalized_error(r_surf, I_fil_1, dl_fil_1, r_fil_1, nn, sg)
loss_fil_2 = LossFunction.normalized_error(r_surf, I_fil_2, dl_fil_2, r_fil_2, nn, sg)
loss_fil_3 = LossFunction.normalized_error(r_surf, I_fil_3, dl_fil_3, r_fil_3, nn, sg)
loss_fil_4 = LossFunction.normalized_error(r_surf, I_fil_4, dl_fil_4, r_fil_4, nn, sg)
loss_fil_5 = LossFunction.normalized_error(r_surf, I_fil_5, dl_fil_5, r_fil_5, nn, sg)
loss_fil_6 = LossFunction.normalized_error(r_surf, I_fil_6, dl_fil_6, r_fil_6, nn, sg)
loss_fil_7 = LossFunction.normalized_error(r_surf, I_fil_7, dl_fil_7, r_fil_7, nn, sg)
loss_fil_8 = LossFunction.normalized_error(r_surf, I_fil_8, dl_fil_8, r_fil_8, nn, sg)
loss_fil_9 = LossFunction.normalized_error(r_surf, I_fil_9, dl_fil_9, r_fil_9, nn, sg)


print(loss_1)
print(loss_3)
print(loss_5)
print(loss_7)
print(loss_9)
print(loss_fil_1)
print(loss_fil_3)
print(loss_fil_5)
print(loss_fil_7)
print(loss_fil_9)

diffs = [0.0, 100 * (loss_fil_1 - loss_1) / (loss_1), 100 * (loss_fil_2 - loss_2) / (loss_2), 100 * (loss_fil_3 - loss_3) / (loss_3), 100 * (loss_fil_4 - loss_4) / (loss_4), 100 * (loss_fil_5 - loss_5) / (loss_5), 100 * (loss_fil_6 - loss_6) / (loss_6), 100 * (loss_fil_7 - loss_7) / (loss_7), 100 * (loss_fil_8 - loss_8) / (loss_8), 100 * (loss_fil_9 - loss_9) / (loss_9)]

plt.loglog(coil_sizes, diffs)
plt.show()

"""
# best fit

power = 4.0

slope = 0.0054 ## NEEDS TO BE CHANGED

x = np.linspace(0.0, 40, 200)
y = slope * (x**power)
"""

#plt.plot(x, y, linestyle='--', color="black")
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 16}
plt.figure(figsize=(7,5.2))
plt.rc('font', **font)
plt.plot(coil_sizes, diffs, linestyle='--', color="grey")
plt.scatter(coil_sizes, diffs)
plt.ylim([0.0,500.0])
plt.ylabel("$\Delta e$ (%)")
plt.xlabel("Coil size (cm)")
plt.savefig("w7x_delta_e.eps")
plt.show()
