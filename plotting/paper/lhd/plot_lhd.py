import pandas as pd
import numpy as np
from mayavi import mlab
from functools import partial
import tables as tb
import sys
sys.path.append("../../../")
from focusadd.coils.CoilSet import CoilSet

def read_lhd_data():
	r_surf = np.load("../../../focusadd/initFiles/lhd/lhd_r_surf.npy")
	nn = np.load("../../../focusadd/initFiles/lhd/lhd_nn_surf.npy")
	sg = np.load("../../../focusadd/initFiles/lhd/lhd_sg_surf.npy")
	fc_init = np.load("../../../focusadd/initFiles/lhd/lhd_fc.npy")
	return r_surf, nn, sg, fc_init

def plot_surface(r):
	x = r[:,:,0]
	y = r[:,:,1]
	z = r[:,:,2]
	p = mlab.mesh(x,y,z,color=(0.8,0.0,0.0))
	return p
def get_coil_data():
	NC = 2
	NS = 96
	NFC = 10
	NFR = 3
	ln = 0.07
	lb = 0.07
	NNR = 2
	NBR = 2
	rc = 2.0
	NR = 0
	return NC, NS, NFC, NFR, ln, lb, NNR, NBR, rc, NR
def get_all_coil_data(filename):
	with tb.open_file(filename, "r") as f:
		coil_data = f.root.metadata[0]
		fc = np.asarray(f.root.coilSeries[:, :, :])
		fr = np.asarray(f.root.rotationSeries[:, :]) # NEEDS TO BE EDITED
		params = (fc, fr)
	return coil_data, params


def compute_r_centroid(NS, fc):
	theta = np.linspace(0, 2 * np.pi, NS + 1)
	NC = fc.shape[1]
	NF = fc.shape[2]
	xc = fc[0]
	yc = fc[1]
	zc = fc[2]
	xs = fc[3]
	ys = fc[4]
	zs = fc[5]
	x = np.zeros((NC, NS + 1))
	y = np.zeros((NC, NS + 1))
	z = np.zeros((NC, NS + 1))
	for m in range(NF):
		arg = m * theta
		carg = np.cos(arg)
		sarg = np.sin(arg)
		x += (
			xc[:, np.newaxis, m] * carg[np.newaxis, :]
			+ xs[:, np.newaxis, m] * sarg[np.newaxis, :]
		)
		y += (
			yc[:, np.newaxis, m] * carg[np.newaxis, :]
			+ ys[:, np.newaxis, m] * sarg[np.newaxis, :]
		)
		z += (
			zc[:, np.newaxis, m] * carg[np.newaxis, :]
			+ zs[:, np.newaxis, m] * sarg[np.newaxis, :]
		)
	return np.concatenate(
		(x[:, :, np.newaxis], y[:, :, np.newaxis], z[:, :, np.newaxis]), axis=2
	)

def plot_coils_centroid(r_coils, color=(0.0, 0.0, 0.8)):
	r_coils = np.concatenate((r_coils[:, :, :, :, :], r_coils[:, 0:1, :, :, :]),axis=1)
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

mlab.clf()
NS = 96
r_surf, nn, sg, fc_init = read_lhd_data()
coil_data, params = get_all_coil_data("../../../tests/lhd/scan/lhd_l4.hdf5")
#coil_data_fb, params_fb = get_all_coil_data("fb_lhd_test.hdf5")
r_init = compute_r_centroid(NS, fc_init)
_, _, r_c, _ = CoilSet.get_outputs(coil_data, params)
#_, _, r_fb, _ = CoilSet.get_outputs(coil_data_fb, params_fb)
mlab.options.offscreen = True
mlab.figure(size=(2400,2400), bgcolor=(1,1,1))

p = plot_surface(r_surf)
#p = plot_coils_centroid(r_init,color=(0.0,0.0,0.8))
p = plot_coils(r_c,color=(0.0,0.0,0.8))
#p = plot_coils(r_fb,color=(0.0,0.0,0.8))
mlab.view(azimuth=0.0, elevation=180, distance=20.0, focalpoint=(0.0,0.0,0.0), roll=None, reset_roll=True, figure=None)
mlab.savefig('lhd_coils.eps', figure=mlab.gcf())

