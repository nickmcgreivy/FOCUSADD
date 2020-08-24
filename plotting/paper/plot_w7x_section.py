from mayavi import mlab
import numpy as np
import tables as tb
import sys
sys.path.append("../../")
from focusadd.coils.CoilSet import CoilSet
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
		p = mlab.plot3d(r_coils[ic,:,0], r_coils[ic,:,1], r_coils[ic,:,2], tube_radius=0.01, color = color)#, line_width = 0.01,)
	return p


def plot_coils(r_coils, color=(0.0, 0.0, 0.8)):
	r_coils = np.concatenate((r_coils[:, :, :, :, :], r_coils[:, 0:1, :, :, :]),axis=1)
	for ic in range(r_coils.shape[0]):
		for n in range(r_coils.shape[2]):
			for b in range(r_coils.shape[3]):
				p = mlab.plot3d(r_coils[ic,:,n,b,0], r_coils[ic,:,n,b,1], r_coils[ic,:,n,b,2], tube_radius=0.01, color = color)#, line_width = 0.01,)
	return p

coil_data_9, params_9 = get_all_coil_data("../../tests/w7x/scan2/w7x_l9.hdf5") # 0.06
I_9, dl_9, r_9, _ = CoilSet.get_outputs(coil_data_9, params_9)
centroid_9 = CoilSet.get_r_centroid(coil_data_9, params_9)

r_surf, nn, sg, _ = read_w7x_data()
mlab.options.offscreen = True
mlab.figure(size=(2400,2400), bgcolor=(1,1,1))
p = plot_surface(r_surf[146:])
p = plot_surface(r_surf[0:19])
p= plot_coils(r_9[0:5], color=(0.0,0.0,0.8))
mlab.view(azimuth=200, elevation=110, distance=9, focalpoint=(5.3,1.5,0.1), roll=None, reset_roll=True)
mlab.savefig('w7x_section.png', figure=mlab.gcf())
#mlab.savefig('w7x.pdf', figure=mlab.gcf())
#mlab.savefig('w7x.eps', figure=mlab.gcf())
mlab.clf()
