import numpy as np
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

radii = np.linspace(0.0,1.2,4)

start = time.time()

rs, zs = pc.getPoincarePoints(200,0.0,radii)

end = time.time()
print(end-start)
print(rs)
print(zs)

plt.plot(rs,zs,'ko', markersize=1)
plt.show()
