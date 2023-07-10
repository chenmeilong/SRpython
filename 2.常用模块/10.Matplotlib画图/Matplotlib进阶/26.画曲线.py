import matplotlib.pyplot as plt
import numpy as np
T = np.array([6, 7, 8, 9, 10, 11, 12])
power = np.array([1.53E+03, 5.92E+02, 2.04E+02, 7.24E+01, 2.72E+01, 1.10E+01, 4.70E+00])

from scipy.interpolate import make_interp_spline
xnew = np.linspace(T.min(),T.max(),300) #300 represents number of points to make between T.min and T.max
power_smooth =make_interp_spline(T, power)(xnew)
plt.plot(xnew,power_smooth)
plt.show()