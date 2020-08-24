from mayavi import mlab
import numpy as np
import tables as tb
import sys
sys.path.append("../../focusadd")
from coils.CoilSet import CoilSet
from lossFunctions.LossFunction import LossFunction
from lossFunctions.DefaultLoss import default_loss, lhd_saddle_B
import matplotlib.pyplot as plt
import pdb

def read_lhd_data():
	r_surf = np.load("../../focusadd/initFiles/lhd/lhd_r_surf.npy")
	nn = np.load("../../focusadd/initFiles/lhd/lhd_nn_surf.npy")
	sg = np.load("../../focusadd/initFiles/lhd/lhd_sg_surf.npy")
	fc_init = np.load("../../focusadd/initFiles/lhd/lhd_fc.npy")
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

I = np.load("../../focusadd/initFiles/lhd/lhd_I_c.npy")

coil_data_fil, params_fil = get_all_coil_data("../../tests/lhd/scan/lhd_l0.hdf5") # fil
I_fil, dl_fil, r_fil, _ = CoilSet.get_outputs(coil_data_fil, params_fil)
centroid_fil = CoilSet.get_r_centroid(coil_data_fil, params_fil)

coil_data_1, params_1 = get_all_coil_data("../../tests/lhd/scan/lhd_l1.hdf5") # 0.025
I_1, dl_1, r_1, _ = CoilSet.get_outputs(coil_data_1, params_1)
centroid_1 = CoilSet.get_r_centroid(coil_data_1, params_1)

coil_data_2, params_2 = get_all_coil_data("../../tests/lhd/scan/lhd_l2.hdf5") # 0.05
I_2, dl_2, r_2, _ = CoilSet.get_outputs(coil_data_2, params_2, I = I)
centroid_2 = CoilSet.get_r_centroid(coil_data_2, params_2)

coil_data_3, params_3 = get_all_coil_data("../../tests/lhd/scan/lhd_l3.hdf5") # 0.075
I_3, dl_3, r_3, _ = CoilSet.get_outputs(coil_data_3, params_3, I = I)
centroid_3 = CoilSet.get_r_centroid(coil_data_3, params_3)

coil_data_4, params_4 = get_all_coil_data("../../tests/lhd/scan/lhd_l4.hdf5") # 0.10
I_4, dl_4, r_4, _ = CoilSet.get_outputs(coil_data_4, params_4, I = I)
centroid_4 = CoilSet.get_r_centroid(coil_data_4, params_4)

coil_data_5, params_5 = get_all_coil_data("../../tests/lhd/scan/lhd_l5.hdf5") # 0.125
I_5, dl_5, r_5, _ = CoilSet.get_outputs(coil_data_5, params_5, I = I)
centroid_5 = CoilSet.get_r_centroid(coil_data_5, params_5)

coil_data_6, params_6 = get_all_coil_data("../../tests/lhd/scan/lhd_l6.hdf5") # 0.15
I_6, dl_6, r_6, _ = CoilSet.get_outputs(coil_data_6, params_6, I = I)
centroid_6 = CoilSet.get_r_centroid(coil_data_6, params_6)

coil_data_7, params_7 = get_all_coil_data("../../tests/lhd/scan/lhd_l7.hdf5") # 0.175
I_7, dl_7, r_7, _ = CoilSet.get_outputs(coil_data_7, params_7, I = I)
centroid_7 = CoilSet.get_r_centroid(coil_data_7, params_7)

coil_data_8, params_8 = get_all_coil_data("../../tests/lhd/scan/lhd_l8.hdf5") # 0.2
I_8, dl_8, r_8, _ = CoilSet.get_outputs(coil_data_8, params_8, I = I)
centroid_8 = CoilSet.get_r_centroid(coil_data_8, params_8)




r_surf, nn, sg, _ = read_lhd_data()


norm_1 = np.mean(np.abs(np.reshape(centroid_fil - centroid_1,-1)))
norm_2 = np.mean(np.abs(np.reshape(centroid_fil - centroid_2,-1)))
norm_3 = np.mean(np.abs(np.reshape(centroid_fil - centroid_3,-1)))
norm_4 = np.mean(np.abs(np.reshape(centroid_fil - centroid_4,-1)))
norm_5 = np.mean(np.abs(np.reshape(centroid_fil - centroid_5,-1)))
norm_6 = np.mean(np.abs(np.reshape(centroid_fil - centroid_6,-1)))
norm_7 = np.mean(np.abs(np.reshape(centroid_fil - centroid_7,-1)))
norm_8 = np.mean(np.abs(np.reshape(centroid_fil - centroid_8,-1)))


coil_sizes = [0.0, 0.025 * 300, 0.05 * 300, 0.075 * 300, 0.1 * 300, 0.125 * 300, 0.15 * 300, 0.175 * 300, .2 * 300]
norms = [0.0, norm_1 * 1000, norm_2 * 1000, norm_3 * 1000, norm_4 * 1000, norm_5 * 1000, norm_6 * 1000, norm_7 * 1000, norm_8 * 1000]

print(norm_4*1000)

plt.plot(coil_sizes, norms, linestyle='--', color="grey")
plt.scatter(coil_sizes, norms)
plt.ylim([0.0,12.0])
plt.ylabel("$||\Delta r||_1$ (mm)")
plt.xlabel("Coil size (cm)")
plt.show()

surface_data = (r_surf, nn, sg)
B_extern = lhd_saddle_B(surface_data, 256)


loss_1 = LossFunction.quadratic_flux(r_surf, I_1, dl_1, r_1, nn, sg, B_extern = B_extern)
loss_2 = LossFunction.quadratic_flux(r_surf, I_2, dl_2, r_2, nn, sg, B_extern = B_extern)
loss_3 = LossFunction.quadratic_flux(r_surf, I_3, dl_3, r_3, nn, sg, B_extern = B_extern)
loss_4 = LossFunction.quadratic_flux(r_surf, I_4, dl_4, r_4, nn, sg, B_extern = B_extern)
loss_5 = LossFunction.quadratic_flux(r_surf, I_5, dl_5, r_5, nn, sg, B_extern = B_extern)
loss_6 = LossFunction.quadratic_flux(r_surf, I_6, dl_6, r_6, nn, sg, B_extern = B_extern)
loss_7 = LossFunction.quadratic_flux(r_surf, I_7, dl_7, r_7, nn, sg, B_extern = B_extern)
loss_8 = LossFunction.quadratic_flux(r_surf, I_8, dl_8, r_8, nn, sg, B_extern = B_extern)
I_fil_1, dl_fil_1, r_fil_1, _ = CoilSet.get_outputs(coil_data_1, params_fil, I = I)
I_fil_2, dl_fil_2, r_fil_2, _ = CoilSet.get_outputs(coil_data_2, params_fil, I = I)
I_fil_3, dl_fil_3, r_fil_3, _ = CoilSet.get_outputs(coil_data_3, params_fil, I = I)
I_fil_4, dl_fil_4, r_fil_4, _ = CoilSet.get_outputs(coil_data_4, params_fil, I = I)
I_fil_5, dl_fil_5, r_fil_5, _ = CoilSet.get_outputs(coil_data_5, params_fil, I = I)
I_fil_6, dl_fil_6, r_fil_6, _ = CoilSet.get_outputs(coil_data_6, params_fil, I = I)
I_fil_7, dl_fil_7, r_fil_7, _ = CoilSet.get_outputs(coil_data_7, params_fil, I = I)
I_fil_8, dl_fil_8, r_fil_8, _ = CoilSet.get_outputs(coil_data_8, params_fil, I = I)
loss_fil_1 = LossFunction.quadratic_flux(r_surf, I_fil_1, dl_fil_1, r_fil_1, nn, sg, B_extern = B_extern)
loss_fil_2 = LossFunction.quadratic_flux(r_surf, I_fil_2, dl_fil_2, r_fil_2, nn, sg, B_extern = B_extern)
loss_fil_3 = LossFunction.quadratic_flux(r_surf, I_fil_3, dl_fil_3, r_fil_3, nn, sg, B_extern = B_extern)
loss_fil_4 = LossFunction.quadratic_flux(r_surf, I_fil_4, dl_fil_4, r_fil_4, nn, sg, B_extern = B_extern)
loss_fil_5 = LossFunction.quadratic_flux(r_surf, I_fil_5, dl_fil_5, r_fil_5, nn, sg, B_extern = B_extern)
loss_fil_6 = LossFunction.quadratic_flux(r_surf, I_fil_6, dl_fil_6, r_fil_6, nn, sg, B_extern = B_extern)
loss_fil_7 = LossFunction.quadratic_flux(r_surf, I_fil_7, dl_fil_7, r_fil_7, nn, sg, B_extern = B_extern)
loss_fil_8 = LossFunction.quadratic_flux(r_surf, I_fil_8, dl_fil_8, r_fil_8, nn, sg, B_extern = B_extern)


diffs = [0.0, 100 * (loss_fil_1 - loss_1) / (loss_1), 100 * (loss_fil_2 - loss_2) / (loss_2), 100 * (loss_fil_3 - loss_3) / (loss_3), 100 * (loss_fil_4 - loss_4) / (loss_4), 100 * (loss_fil_5 - loss_5) / (loss_5), 100 * (loss_fil_6 - loss_6) / (loss_6), 100 * (loss_fil_7 - loss_7) / (loss_7), 100 * (loss_fil_8 - loss_8) / (loss_8)]

plt.plot(coil_sizes, diffs, linestyle='--', color="grey")
plt.scatter(coil_sizes, diffs)
plt.ylim([0.0,20.0])
plt.ylabel("$\Delta f$ (%)")
plt.xlabel("Coil size (cm)")
plt.show()


pdb.set_trace()