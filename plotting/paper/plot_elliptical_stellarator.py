
from mayavi import mlab
import numpy as np
import tables as tb
import sys
sys.path.append("../../")
from focusadd.coils.CoilSet import CoilSet
from focusadd.surface.Surface import Surface 



def plot_surface(r):
	r = np.concatenate((r[:, :], r[:, 0:1]),axis=1)
	r = np.concatenate((r[:, :], r[0:1, :]),axis=0)
	x = r[:,:,0]
	y = r[:,:,1]
	z = r[:,:,2]
	p = mlab.mesh(x,y,z,color=(0.8,0.0,0.0))
	return p
	
def get_coil_data():
	NC = 50
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


def plot_coils_centroid(r_coils, color=(0.0, 0.0, 0.8)):
	r_coils = np.concatenate((r_coils[:, :, :], r_coils[:, 0:1, :]),axis=1)
	for ic in range(r_coils.shape[0]):
		p = mlab.plot3d(r_coils[ic,:,0], r_coils[ic,:,1], r_coils[ic,:,2], tube_radius=0.02, color = color)#, line_width = 0.01,)
	return p
def plot_coils(r_coils, color=(0.0, 0.0, 0.8)):
	r_coils = np.concatenate((r_coils[:, :, :, :, :], r_coils[:, 0:1, :, :, :]),axis=1)
	for ic in range(r_coils.shape[0]):
		for n in range(r_coils.shape[2]):
			for b in range(r_coils.shape[3]):
				p = mlab.plot3d(r_coils[ic,:,n,b,0], r_coils[ic,:,n,b,1], r_coils[ic,:,n,b,2], tube_radius=0.003, color = color)#, line_width = 0.01,)
	return p

surface = Surface("../../focusadd/initFiles/axes/ellipticalAxis4Rotate.txt", 128, 32, 1.0)
r_surf = surface.get_r_central()

coil_data_fb, params_fb = CoilSet.get_initial_data(surface, input_file="../../tests/postresaxis/triple_comparison/fb.hdf5")
_, _, r_fb, _ = CoilSet.get_outputs(coil_data_fb, params_fb, I_input = None)
centroid_fb = CoilSet.get_r_centroid(coil_data_fb, params_fb)

mlab.options.offscreen = True
mlab.figure(size=(2400,1600), bgcolor=(1,1,1))
p = plot_surface(r_surf)
#r_c = compute_r_centroid(NS, fc_init)
#p = plot_coils(r_fil, color=(0.8,0.0,0.0))
p = plot_coils(r_fb, color=(0.0,0.0,0.8))
mlab.view(azimuth=0, elevation=0, distance=4.0, focalpoint=(0.0,0.0,0.0), reset_roll=True)
mlab.savefig('elliptical_stellarator.png', figure=mlab.gcf())
mlab.clf()
