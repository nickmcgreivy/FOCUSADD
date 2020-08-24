from mayavi import mlab
import numpy as np
import tables as tb
import sys
sys.path.append("../../")
from focusadd.coils.CoilSet import CoilSet
from focusadd.lossFunctions.LossFunction import LossFunction
import matplotlib.pyplot as plt

def read_w7x_data():
	r_surf = np.load("../../focusadd/initFiles/w7x/w7x_r_surf.npy")
	nn = np.load("../../focusadd/initFiles/w7x/w7x_nn_surf.npy")
	sg = np.load("../../focusadd/initFiles/w7x/w7x_sg_surf.npy")
	fc_init = np.load("../../focusadd/initFiles/w7x/w7x_fc.npy")
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

I = np.ones(50)

coil_data_fil, params_fil = get_all_coil_data("../../tests/w7x/scan2/w7x_l0.hdf5") # fil
I_fil, dl_fil, r_fil, _ = CoilSet.get_outputs(coil_data_fil, params_fil)
centroid_fil = CoilSet.get_r_centroid(coil_data_fil, params_fil)

coil_data_1, params_1 = get_all_coil_data("../../tests/w7x/scan2/w7x_l1.hdf5") # 0.035
I_1, dl_1, r_1, _ = CoilSet.get_outputs(coil_data_1, params_1)
centroid_1 = CoilSet.get_r_centroid(coil_data_1, params_1)

coil_data_2, params_2 = get_all_coil_data("../../tests/w7x/scan2/w7x_l2.hdf5") # 0.07
I_2, dl_2, r_2, _ = CoilSet.get_outputs(coil_data_2, params_2)
centroid_2 = CoilSet.get_r_centroid(coil_data_2, params_2)

coil_data_3, params_3 = get_all_coil_data("../../tests/w7x/scan2/w7x_l3.hdf5") # 0.14
I_3, dl_3, r_3, _ = CoilSet.get_outputs(coil_data_3, params_3)
centroid_3 = CoilSet.get_r_centroid(coil_data_3, params_3)

coil_data_5, params_5 = get_all_coil_data("../../tests/w7x/scan2/w7x_l5.hdf5") # 0.105
I_5, dl_5, r_5, _ = CoilSet.get_outputs(coil_data_5, params_5)
centroid_5 = CoilSet.get_r_centroid(coil_data_5, params_5)

coil_data_6, params_6 = get_all_coil_data("../../tests/w7x/scan2/w7x_l6.hdf5") # 0.0175
I_6, dl_6, r_6, _ = CoilSet.get_outputs(coil_data_6, params_6)
centroid_6 = CoilSet.get_r_centroid(coil_data_6, params_6)

coil_data_7, params_7 = get_all_coil_data("../../tests/w7x/scan2/w7x_l7.hdf5") # 0.525
I_7, dl_7, r_7, _ = CoilSet.get_outputs(coil_data_7, params_7)
centroid_7 = CoilSet.get_r_centroid(coil_data_7, params_7)

coil_data_8, params_8 = get_all_coil_data("../../tests/w7x/scan2/w7x_l8.hdf5") # 0.0875
I_8, dl_8, r_8, _ = CoilSet.get_outputs(coil_data_8, params_8)
centroid_8 = CoilSet.get_r_centroid(coil_data_8, params_8)

coil_data_9, params_9 = get_all_coil_data("../../tests/w7x/scan2/w7x_l9.hdf5") # 0.06
I_9, dl_9, r_9, _ = CoilSet.get_outputs(coil_data_9, params_9)
centroid_9 = CoilSet.get_r_centroid(coil_data_9, params_9)




r_surf, nn, sg, _ = read_w7x_data()
mlab.options.offscreen = True
mlab.figure(size=(1600,1200), bgcolor=(1,1,1))

#p = plot_surface(r_surf)

L = 0
U = 5
p = plot_coils(r_3, color=(0.8,0.0,0.0))
#mlab.view(azimuth=30.0, elevation=90, distance=6.0, focalpoint=(5.5,1.8,0.0), roll=None, reset_roll=True)

mlab.savefig('w7x_c3.png', figure=mlab.gcf())
mlab.clf()

norm_1 = np.mean(np.abs(np.reshape(centroid_fil - centroid_1,-1)))
norm_2 = np.mean(np.abs(np.reshape(centroid_fil - centroid_2,-1)))
norm_3 = np.mean(np.abs(np.reshape(centroid_fil - centroid_3,-1)))
norm_5 = np.mean(np.abs(np.reshape(centroid_fil - centroid_5,-1)))
norm_6 = np.mean(np.abs(np.reshape(centroid_fil - centroid_6,-1)))
norm_7 = np.mean(np.abs(np.reshape(centroid_fil - centroid_7,-1)))
norm_8 = np.mean(np.abs(np.reshape(centroid_fil - centroid_8,-1)))
norm_9 = np.mean(np.abs(np.reshape(centroid_fil - centroid_9,-1)))

coil_sizes = [0.0, 0.0525 * 100, .105 * 100, 0.1575 * 100, .18 * 100, .21 * 100, 0.2625 * 100, 0.105 * 3 * 100, .42 * 100]
norms = [0.0, norm_6 * 1000, norm_1 * 1000, norm_7 * 1000, norm_9 * 1000, norm_2 * 1000, norm_8 * 1000, norm_5 * 1000, norm_3 * 1000]

plt.loglog(coil_sizes, norms, linestyle='--', color="grey")
plt.scatter(coil_sizes, norms)
plt.ylim([0.0,12.0])
plt.ylabel("$||\Delta r||_1$ (mm)")
plt.xlabel("Coil size (cm)")
plt.show()



loss_1 = LossFunction.quadratic_flux(r_surf, I_1, dl_1, r_1, nn, sg)
loss_2 = LossFunction.quadratic_flux(r_surf, I_2, dl_2, r_2, nn, sg)
loss_3 = LossFunction.quadratic_flux(r_surf, I_3, dl_3, r_3, nn, sg)
loss_5 = LossFunction.quadratic_flux(r_surf, I_5, dl_5, r_5, nn, sg)
loss_6 = LossFunction.quadratic_flux(r_surf, I_6, dl_6, r_6, nn, sg)
loss_7 = LossFunction.quadratic_flux(r_surf, I_7, dl_7, r_7, nn, sg)
loss_8 = LossFunction.quadratic_flux(r_surf, I_8, dl_8, r_8, nn, sg)
loss_9 = LossFunction.quadratic_flux(r_surf, I_9, dl_9, r_9, nn, sg)
I_fil_1, dl_fil_1, r_fil_1, _ = CoilSet.get_outputs(coil_data_1, params_fil)
I_fil_2, dl_fil_2, r_fil_2, _ = CoilSet.get_outputs(coil_data_2, params_fil)
I_fil_3, dl_fil_3, r_fil_3, _ = CoilSet.get_outputs(coil_data_3, params_fil)
I_fil_5, dl_fil_5, r_fil_5, _ = CoilSet.get_outputs(coil_data_5, params_fil)
I_fil_6, dl_fil_6, r_fil_6, _ = CoilSet.get_outputs(coil_data_6, params_fil)
I_fil_7, dl_fil_7, r_fil_7, _ = CoilSet.get_outputs(coil_data_7, params_fil)
I_fil_8, dl_fil_8, r_fil_8, _ = CoilSet.get_outputs(coil_data_8, params_fil)
I_fil_9, dl_fil_9, r_fil_9, _ = CoilSet.get_outputs(coil_data_9, params_fil)
loss_fil_1 = LossFunction.quadratic_flux(r_surf, I_fil_1, dl_fil_1, r_fil_1, nn, sg)
loss_fil_2 = LossFunction.quadratic_flux(r_surf, I_fil_2, dl_fil_2, r_fil_2, nn, sg)
loss_fil_3 = LossFunction.quadratic_flux(r_surf, I_fil_3, dl_fil_3, r_fil_3, nn, sg)
loss_fil_5 = LossFunction.quadratic_flux(r_surf, I_fil_5, dl_fil_5, r_fil_5, nn, sg)
loss_fil_6 = LossFunction.quadratic_flux(r_surf, I_fil_6, dl_fil_6, r_fil_6, nn, sg)
loss_fil_7 = LossFunction.quadratic_flux(r_surf, I_fil_7, dl_fil_7, r_fil_7, nn, sg)
loss_fil_8 = LossFunction.quadratic_flux(r_surf, I_fil_8, dl_fil_8, r_fil_8, nn, sg)
loss_fil_9 = LossFunction.quadratic_flux(r_surf, I_fil_9, dl_fil_9, r_fil_9, nn, sg)


diffs = [0.0, 100 * (loss_fil_6 - loss_6) / (loss_6), 100 * (loss_fil_1 - loss_1) / (loss_1), 100 * (loss_fil_7 - loss_7) / (loss_7), 100 * (loss_fil_9 - loss_9) / (loss_9), 100 * (loss_fil_2 - loss_2) / (loss_2), 100 * (loss_fil_8 - loss_8) / (loss_8), 100 * (loss_fil_5 - loss_5) / (loss_5), 100 * (loss_fil_3 - loss_3) / (loss_3)]

plt.loglog(coil_sizes, diffs, linestyle='--', color="grey")
plt.scatter(coil_sizes, diffs)
plt.ylim([0.0,1500.0])
plt.ylabel("$\Delta f$ (%)")
plt.xlabel("Coil size (cm)")
plt.show()
