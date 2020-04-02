import jax.numpy as np
import matplotlib.pyplot as plt
import sys
import time
sys.path.append("../..")

from focusadd.surface.readAxis import readAxis
from focusadd.surface.Surface import Surface
from focusadd.coils.CoilSet import CoilSet
from focusadd.poincare.Poincare import Poincare

surface = Surface("../initFiles/axes/defaultAxis.txt", 128, 32, 1.0)
coilSet = CoilSet(surface,input_file="../../tests/validateWithFocus/validateFOCUS.hdf5")

pc = Poincare(coilSet, surface)

radii = np.linspace(0.000,1.2,1)

start = time.time()

sol = pc.getPoincarePoints(6,0.0,radii)

end = time.time()
print(end-start)
print(sol.t)
print(sol.y)

plt.plot(sol.y[0],sol.y[1],'k,', markersize=2)
plt.show()
