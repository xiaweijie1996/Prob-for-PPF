import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, PchipInterpolator, Akima1DInterpolator

# Generate sample data
x = np.linspace(0, 1, 100)
a1, b1, c1 = 10, 3, 1
a2, b2, c2 = 2, 4, 1.2
y1 = a1*x**2 + b1*x + c1
y2 = a2*x**2 + b2*x + c2

y = y1/y2

# PLot y and x
plt.figure(figsize=(8, 6))
plt.plot(x, y, 'b--', label='Data Points')
# plot the lines
plt.plot(x, y1, 'r--', label='y1 = (1*x^2 - 0.5*x + 2)')
plt.plot(x, y2, 'g--', label='y2 = (-1*x^2 + 0.5*x - 1)')
# plot x and y lines

plt.title('Data Points')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.savefig("test/spline_data_points.png")