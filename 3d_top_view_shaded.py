import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

def rastrigin(x):
    # can be applied to all arrays with previous last axis being
    n = x.shape[-1]
    return 10 * n + np.sum(x * x - 10 * np.cos(2 * np.pi * x), axis=-1)

cmap = plt.cm.nipy_spectral
# cmap = plt.cm.jet

fig, ax = plt.subplots()
M = 5.12
# x = np.linspace(-M, M, 1000)
# y = np.linspace(-M, M, 1000)
# X, Y = np.meshgrid(x, y)
# XY = np.moveaxis(np.array([X, Y]), 0, -1)
# Z = rastrigin(XY)
ls = LightSource(azdeg=345, altdeg=45)
XY = np.array(np.meshgrid(np.arange(-M, M, .005),
                          np.arange(-M, M, .005)))
Z = rastrigin(np.moveaxis(XY, 0, -1))
ax.imshow(ls.shade(Z, cmap=cmap, vert_exag=1), extent=(-M, M, -M, M))
plt.show()
