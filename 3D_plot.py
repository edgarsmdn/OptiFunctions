from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

'''
-------------------------------------------------------------------------------
------------------------ 3D PLOTTING ALGORITHM --------------------------------
------------------------ Coded by Edgar Sanchez --------------------------------

Note: Every line with a checkmark requires your attention to make a proper plot
-------------------------------------------------------------------------------
'''
# .............................................................................
#!!! Import the proper function you want to plot
from Opt_Functions import weierstrass

#!!! Change the definition of f with the function you are importing
f = weierstrass

#!!! Set the proper bounds to evaluate the function
bounds = (-0.5, 0.5)

#!!! Write the title of the plot
title = 'Weierstrass function'

#!!! Set where in the z axis to locate the 2D projection (Set it at the global minimum)
projection = 4
# .............................................................................

# Defines figure and axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Dimension of 2 is setted for making the 3D plots
dim = 2

# Sets tile of the plot and ajusts its properties
plt.title(title , y=1.02, fontsize=15)

# Space between lines in the surface grid
space_surf_grid = np.abs(bounds[0]-bounds[1])/500

# Defines ranges for x and y axes 
x = np.arange(bounds[0],bounds[1],space_surf_grid)
y = np.arange(bounds[0],bounds[1],space_surf_grid)

# Makes a matrix of the x and y vectors
X, Y = np.meshgrid(x, y)

# Defines zs values with the function
zs = np.array([f(np.array([x,y])) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)

# Edits axes labels
ax.set_xlabel(r'$\mathit{x_1}$', fontsize=12, labelpad=10)
ax.set_ylabel(r'$\mathit{x_2}$', fontsize=12, labelpad=10)
ax.set_zlabel(r'$\mathit{f\;(x_1,x_2)}$', fontsize=12, labelpad=10)

# Defines surface plot with certain color map
surf = ax.plot_surface(X, Y, Z, cmap=cm.jet)

# Adds a color bar which maps values to colors and edits its properties
fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.13)

# Defines middle ticks to be shown in the plot
a = np.abs(bounds[0] - bounds[1])
b = bounds[0] + a/4
c = bounds[0] + 2*a/4
d = bounds[0] + 3*a/4

# Adds the boundary ticks and he middle ticks that were defined above
ax.set_xticks([bounds[0], b, c, d, bounds[1]])
ax.set_yticks([bounds[0], b, c, d, bounds[1]])

# Makes a 2D projection on the floor of the graph and edits its properties
ax.contour(X, Y, Z, zdir='z', offset=projection, cmap=cm.jet, linewidths=1)

# Edits the graph grid properties
ax.xaxis._axinfo["grid"].update({"linewidth":0.5})
ax.yaxis._axinfo["grid"].update({"linewidth":0.5})
ax.zaxis._axinfo["grid"].update({"linewidth":0.5})

ax.xaxis._axinfo["grid"]['linestyle'] = "--"
ax.yaxis._axinfo["grid"]['linestyle'] = "--"
ax.zaxis._axinfo["grid"]['linestyle'] = "--"

# Shows the plot
plt.show()