from mayavi import mlab
import numpy as np
import tables as tb
import sys
sys.path.append("../../../")
from focusadd.coils.CoilSet import CoilSet
from focusadd.lossFunctions.LossFunction import LossFunction


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

coil_data_fil, params_fil = get_all_coil_data("../../../tests/w7x/scan3/w7x_l0.hdf5")
I_fil, dl_fil, r_fil, _ = CoilSet.get_outputs(coil_data_fil, params_fil)
centroid_fil = CoilSet.get_r_centroid(coil_data_fil, params_fil)
coil_data_fb, params_fb = get_all_coil_data("../../../tests/w7x/scan3/w7x_l3.hdf5") # 0.06 * 3
I_fb, dl_fb, r_fb, _ = CoilSet.get_outputs(coil_data_fb, params_fb)
centroid_fb = CoilSet.get_r_centroid(coil_data_fb, params_fb)
"""
coil_data_rot, params_rot = get_all_coil_data("../../tests/w7x/w7x_rot.hdf5")
_, _, r_rot, _ = CoilSet.get_outputs(coil_data_rot, params_rot)
centroid_rot = CoilSet.get_r_centroid(coil_data_rot, params_rot)
"""

r_surf, nn, sg, _ = read_w7x_data()
mlab.options.offscreen = True
mlab.figure(size=(1600,1200), bgcolor=(1,1,1))
#p = plot_surface(r_surf)
#r_c = compute_r_centroid(NS, fc_init)
L = 0
U = 5
p = plot_coils_centroid(centroid_fil[L:U], color=(0.8,0.0,0.0))
p = plot_coils_centroid(centroid_fb[L:U], color=(0.0,0.0,0.8))
mlab.view(azimuth=30.0, elevation=90, distance=6.0, focalpoint=(5.5,1.8,0.0), roll=None, reset_roll=True)

mlab.savefig('w7x_centroids.png', figure=mlab.gcf())
mlab.savefig('w7x_centroids.eps', figure=mlab.gcf())
mlab.clf()


print(np.mean(np.abs(np.reshape(centroid_fil - centroid_fb,-1))))
print("Finite build loss:")
print(LossFunction.quadratic_flux(r_surf, I_fb, dl_fb, r_fb, nn, sg))
print("Loss with filamentary centroid but finite build:")
I_fil_fb, dl_fil_fb, r_fil_fb, _ = CoilSet.get_outputs(coil_data_fb, params_fil)
print(LossFunction.quadratic_flux(r_surf, I_fil_fb, dl_fil_fb, r_fil_fb, nn, sg))