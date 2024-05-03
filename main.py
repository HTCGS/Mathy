import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from calculus import *


def func(x): return x**2

df = derivative_function(func)
print(df(3))
d2f = derivative_function(df)
print(d2f(3))


a, b = -2, 2
n = 10
s = left_point_rule(func, a, b, n)
print("Left-point sum: {:.4f}".format(s))
s = right_point_rule(func, a, b, n)
print("Right-point sum: {:.4f}".format(s))
s = mid_point_rule(func, a, b, n)
print("Mid-point sum: {:.4f}".format(s))
s = trapezoidal_rule(func, a, b, n)
print("Trapezoidal rule: {:.4f}".format(s))
s = simpsons_rule(func, a, b, n)
print("Simpson's rule: {:.4f}".format(s))
# l = arc_length(func, 0, 2, n)
# print("Arc length: {:.4f}".format(l))
# av = average_value(func, a, b, n)
# print("Average value: {:.4f}".format(av))
# area = surface_area_of_revolution_about_y(func, 0, b, n)
# print("Surface area about y-axis: {:.4f}".format(area))
# area = surface_area_of_revolution_about_x(func, 0, b, n)
# print("Surface area about x-axis: {:.4f}".format(area))
# volume = volume_of_revolution(lambda x: 2*x - x**2, 0, 2, n)
# print("Volume about x-axis: {:.4f}".format(volume))
# volume = volume_of_revolution_between_curves(lambda x: x+4, lambda x: 2*x+1, 0, 1, n)
# print("Volume about x-axis: {:.4f}".format(volume))
# moments = moments_of_system(lambda x: x**4, lambda x: 0, 0, 2, n)
# Mx, My = moments
# print("Moments of the system: Mx= {:.2f} ; My= {:.2f}".format(Mx, My))
# center = center_of_mass_between_curves(lambda x: x**4, lambda x: 0, 0, 2, n)
# x, y = center
# print("Center of mass of the system: x= {:.2f} ; y= {:.2f}".format(x, y))

# print(limit(lambda x: (x**2) / (x**2 * 3), 999999999))
# print(series_root_test(lambda x: (x**2) / (x**2 * 3)))
# print(series_ratio_test(lambda x: (x**2) / (x**2 * 3)))
# print(series_integral_test(lambda x: (x**2) / (x**2 * 3)))

# d = distance_between_point_and_line([-1, 1], 3, 4, -12)
# print("Distance between point and line = {:.2f}".format(d))
# d = distance_between_point_and_plane([10, 20, -30], 8, 0, 6, 15)
# print("Distance between point and plane = {:.2f}".format(d))
# d = distance_between_parallel_planes(3, 1, -4, -11, -34)
# print("Distance between parallel planes = {:.2f}".format(d))
# a = angle_between_planes([2, -3, 0], [1, 3, -7])
# print("Angle between planes = {:.2f}".format(np.rad2deg(a)))

# x = np.linspace(0, 5, 6)
# y = np.linspace(0, 5, 6)
# z = np.linspace(0, 3, 4)
# xx, yy, zz = np.meshgrid(x, y, z)

# dx = 0.01
# x = np.arange(-np.pi, np.pi, dx)
# y = np.arange(-np.pi, np.pi, dx)
# z = np.arange(-np.pi, np.pi, dx)
# xx, yy = np.meshgrid(x, y)

# z = np.sqrt(6*x*y - 2*x**2 - x*y**2 + 3)
# z = xx**3*yy**2
# z = xx**3 - 2*yy

# print(surface_integral(lambda x, y, z: 3*x+2*z, lambda y, z: 4/3+1/3*y-2/3*z, [-4, 0],
#                        lambda y: 0, lambda y: (y+4)/2,  axis=0))

# print(surface_integral(lambda x, y, z: 6*x*y, lambda y, z: 1-y-z, [0, 1],
#                        lambda y: 0, lambda y: 1-y,  axis=0))

# print(gradient_vector_3D(lambda x, y, z: x+2*y+z-2, [0, 0, 2]))


# print(f(0, 1, 0) - f(0, 0, 0))


# dzdx = partial_derivative_2D_by_x(z, [0, 0])
# dzdy = partial_derivative_2D_by_y(z, [0, 0])

# denominator = np.sqrt(dzdx**2 + dzdy**2 + 1)
# i = -dzdx / denominator
# j = -dzdy / denominator
# k = 1 / denominator

# print(i, j, k)

# print(surface_integral(lambda x, y, z: (x+z)*i + (x+3*y)*j + y*k, z, [0, 2], lambda x: 0, lambda x: 1-0.5*x, proj="xy"))
# print(surface_integral(lambda x, y, z: (x+z)*i + (x+3*y)*j + y*k,
#                        z, [0, 1], lambda y: 0, lambda y: 2-2*y, proj="xy", axis=1))
# print(double_integral(lambda x, y: 2*x+5*y+2, [0, 1], lambda y: 0, lambda y: 2-2*y, axis=1))

# print(surface_integral_of_2form(lambda x, y, z: x+z, lambda x, y, z: x+3*y, lambda x, y, z: y, lambda x, y: 2-x-2*y,
#                                 [0, 0, 2], [0, 1], lambda y: 0, lambda y: 2-2*y, axis=1))

# print(surface_integral_of_2form(lambda x, y, z: x*y*z, lambda x, y, z: y, lambda x, y, z: 0, lambda x, y: 1-y**2,
#                                 [-1, 1], lambda x: 0, lambda x: 3, axis=1))


# print(surface_integral_of_2form(lambda x, y, z: x-3*y+6*z, lambda x, y, z: 0, lambda x, y, z: 0, lambda y, z: y+2*z-4,
#                                 [0, 2], lambda z: 0, lambda z: 4-2*z, axis=1))
# print(surface_integral_of_2form(lambda t, r, z: r*np.cos(t) + r*np.cos(t)*z, lambda t, r, z: r, lambda t, r, z: z-r**2*np.cos(t)**2,
#                                 lambda t, r: np.sqrt(4-r**2), [0, 2*np.pi], lambda r: 0, lambda r: 2))

# print(surface_integral(lambda t, r, z: 2*r, lambda t, r: np.sqrt(4-r**2),
#                        [0, 2*np.pi], lambda t: 0, lambda t: 2))


# print(surface_integral(lambda x, y, z: x+y+z, lambda x, y: 1 - x/4-y/2, [0, 2], lambda y: 0, lambda y: 4-2*y, axis=1))

# print(double_integral(lambda y, z: (1+z)*np.sqrt(4-y**2-z**2), [-2, 2], lambda y: 0, lambda y: np.sqrt(4-y**2)))
# print(double_integral(lambda x, z: 2*np.sqrt(4-x**2-z**2), [-2, 2], lambda x: 0, lambda x: np.sqrt(4-x**2)))
# print(double_integral(lambda x, y: np.sqrt(4-x**2-y**2)-x**2, [-2, 2], lambda x: 0, lambda x: np.sqrt(4-x**2)))

# print(surface_integral_greens_rule(lambda x, y, z: x*z, lambda x, y, z: 1, lambda x, y, z: 2*y, [0, 2],
#                                    lambda z: 0, lambda z: 2-z, lambda x, z: 0, lambda x, z: 1-0.5*x-0.5*z, axis="zxy"))
# print(double_integral(lambda x,y:4/np.sqrt(4-x**2-y**2),[-2, 2], lambda t: -np.sqrt(4-t**2), lambda t: np.sqrt(4-t**2), axis=0))
# print(double_integral(lambda r, t: (4*r)/np.sqrt(4-r**2),
#                       [0, 2*np.pi], lambda t: 0, lambda t: 2, axis=1, xn=100, yn=100))


# print(surface_integral_of_2form(lambda x, y, z: x+x*z, lambda x, y, z: y, lambda x, y, z: z-x**2, lambda x, y: np.sqrt(4-x**2-y**2),
#                                 [-2, 2], lambda t: -np.sqrt(4-t**2), lambda t: np.sqrt(4-t**2), axis=0, xn=10, yn=10))
# print(double_integral(lambda r, t: (2*r * 2) / np.sqrt(4-r**2),
#                       [0, 2*np.pi], lambda t: 0, lambda t: 2, axis=1, xn=10, yn=10))
# print(surface_integral(lambda r, t, z: 2*r, lambda r, t: np.sqrt(4 - r**2),
#                        [0, 2*np.pi], lambda t: 0, lambda t: 2, axis=1, xn=10, yn=10)) !!!!

# print(surface_integral_of_2form(lambda r, t, z: r*np.cos(t)+r*np.cos(t)*z,
#                                 lambda r, t, z: r*np.sin(t),
#                                 lambda r, t, z: z-r*np.cos(t)**2,
#                                 lambda r, t: np.sqrt(4-r**2),
#                                 [0, 2*np.pi], lambda t: 0, lambda t: 2, axis=1, xn=100, yn=100))
# print(surface_integral_of_2form(lambda x, y, z: x+x*z, lambda x, y, z: y, lambda x, y, z: z-x**2, lambda r, t: np.sqrt(4-r**2),
#                                 [0, 2*np.pi], lambda t: 0, lambda t: 2, axis=1))
# print(surface_integral_of_2form(lambda x, y, z: x+x*z, lambda x, y, z: y, lambda x, y, z: z-x**2, lambda r, t: np.sqrt(4-r**2),
#                                 [0, 2*np.pi], lambda t: 0, lambda t: 2, axis=1))

# print(surface_integral_greens_rule(lambda x, y, z: x+x*z, lambda x, y, z: y, lambda x, y, z: z-x**2, [0, 2*np.pi],
#                                    lambda t: 0, lambda t: np.pi, lambda phi, t: 0, lambda phi, t: 2, axis="yzx"))
# print(surface_integral_of_2form(lambda r, t, phi:  r*np.sin(t)*np.cos(phi)+r*np.sin(t)*np.cos(phi)*r*np.cos(t),
#                                 lambda r, t, phi:  r*np.sin(t)*np.sin(phi),
#                                 lambda r, t, phi:  r*np.cos(t) - r*np.sin(t)*np.cos(phi)**2,
#                                 lambda r, t: 2,
#                                 [0, 2*np.pi], lambda t: 0, lambda t: 2, axis=1))
# print(surface_integral_greens_rule(lambda x, y, z: convert_to_spherical_function(lambda x1, y1, z1: x+x*z),
#                                    lambda x, y, z: convert_to_spherical_function(lambda x1, y1, z1: y),
#                                    lambda x, y, z: convert_to_spherical_function(lambda x1, y1, z1: z-x**2),
#                                    [0, 2*np.pi],
#                                    lambda t: 0, lambda t: np.pi, lambda phi, t: 0, lambda phi, t: 2, axis="yzx"))

# print(surface_integral(lambda x, y, z: 6*x*y, lambda y, z: 1-y-z,
#                        [0, 1], lambda y: 0, lambda y: 1-y, proj="yz", axis=1))

# dzdx, dzdy = partial_derivatives_by_points(z, dx)
# dzdxx, dzdxy = partial_derivatives_by_points(dzdx, dx)
# dzdyx, dzdyy = partial_derivatives_by_points(dzdy, dx)


# x = np.arange(0, 10, 1)
# y = np.arange(0, 10, 1)
# # x,y = np.meshgrid(x,y)
# u = np.cos(x)
# v = np.sin(y)

# plt.figure()
# plt.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=1)
# plt.xlim(-5, 20)
# plt.ylim(-5, 20)
# plt.grid()
# plt.show()

# strike = 100
# premium = 5
# price = np.arange(80, 120, 1)
# call_holder = np.maximum(price - strike, 0) - premium
# put_holder = np.maximum(strike - price, 0) - premium

# call_plus_put = call_holder + put_holder
# call_writer = -call_holder
# put_writer = -put_holder

# plt.figure()
# plt.plot(price, call_holder)
# plt.plot(price, put_holder)
# plt.plot(price, call_plus_put)
# plt.plot(price, call_writer)
# plt.legend(["Call option", "Putt option"])
# plt.show()

# stock = pd.read_json("G:\MarketData\StockData\BSQR.json")
# volume = stock["Volume"][:-50]
# plt.figure()
# plt.plot(volume)
# plt.show()

# y = x**3
# u = xx**3+yy**2-2*zz

# dudx = derivative_by_points(u[0, :, 0], 0.01)
# dudy = derivative_by_points(u[:, 0, 0], 0.01)
# dudz = derivative_by_points(u[0][0], 0.01)
# plt.figure()
# # plt.plot(x, y)
# plt.plot(x[1:-1], dzdx[0])
# plt.plot(y[1:-1], dzdy[:, 0])
# plt.plot(x[2:-2], dzdxx[0])
# plt.plot(y[1:-1], dzdxy[:, 0])
# plt.plot(x[1:-1], dzdyx[0])
# plt.plot(y[2:-2], dzdyy[:, 0])
# plt.plot(x[1:-1], dudx)
# plt.plot(y[1:-1], dudy)
# plt.plot(z[1:-1], dudz)
# fig = plt.figure()
# ax = plt.axes(projection="3d")
# ax.plot_surface(xx, yy, z, cmap='viridis', edgecolor='none')
# ax.plot_surface(xx, yy, z_tangent, cmap='viridis', edgecolor='none')
# ax.plot_surface(xx, yy, z_normal, cmap='viridis', edgecolor='none')
# plt.grid()
# plt.show()

# grad = gradient_vector(lambda x, y: x - 2*y, [0, 0])
# print(grad)
# dxdl = directional_derivative(lambda x, y, z: np.sin(x+2*y) + np.sqrt(x*y*z), [np.pi / 2, 3*np.pi/2, 2], [4, 3, 0])
# dxdl = directional_derivative(lambda x, y: x**2 - 2*y, [0, 0], grad)
# print(dxdl)

# grad_proj = projection_on_vector(grad, [4, 3, 0])
# print(grad_proj)

# plt.figure()
# plt.plot(x, y)
# plt.quiver([0, 0], [0, 0], grad[0], grad[1], angles='xy', scale_units='xy', scale=1)
# plt.quiver([0, 0], [0, 0], dxdl * 1, dxdl*1, angles='xy', scale_units='xy', scale=1)
# # plt.plot(x[-1][1:-1], dzdx)
# # plt.plot(x[-1][1:-1], dzdy)
# plt.grid()
# plt.show()

# y = x + 1
# y_series = fourier_series(lambda x: x + 1, np.pi, 5, x)

# plt.figure()
# plt.plot(x, y)
# plt.plot(x, y_series)
# plt.show()

# u = np.linspace(-2, 2, 50)
# v = np.linspace(-2, 2, 50)
# u, v = np.meshgrid(u, v)

# x = np.cos(u) * np.sin(v)
# y = np.sin(u) * np.sin(v)
# z = np.cos(v) * np.ones_like(u)

# z = np.sqrt((1 + (x**2) + (y**2)))

# x, z = np.meshgrid(x, z)
# z = np.sin(np.sqrt(x ** 2 + y ** 2))
# z = np.sqrt(x**2 + y**2)
# z = x**2 + y**2
# z = np.sqrt(1 - y**2)
# y = np.sqrt(1 - x**2)
# y = 1/x
# z = x**2 - y**2
# z = np.sqrt(x**2 + y**2 - 1)
# z = np.sqrt(1+x**2 + y**2)

# u, v = np.meshgrid(x, y)
# x = np.sqrt(1+u**2) * np.cos(v)
# y = np.sqrt(1+u**2) * np.sin(v)
# z = u

# z = (x**2 - y**3) / (x**2 + y**2)

# x0 = np.ones(50) * 0
# ftox0 = (x0**2 - v**3) / (x0**2 + v**2)

# f1x = np.linspace(-2, 2, 100)
# f1y = np.linspace(-2, 2, 100)
# f1x, f1y = np.meshgrid(f1x, f1y)

# fig = plt.figure()
# ax = plt.axes(projection="3d")
# ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none')
# ax.scatter(x, y, z, color="m")
# ax.plot_wireframe(x, y, ftox0)
# ax.plot(x0, v, ftox0)
# ax.plot_surface(x, -y, z, cmap='viridis', edgecolor='none')
# ax.scatter(x, y, z, cmap='viridis', edgecolor='none')
# plt.show()

# point = [-1, -3]
# point2 = [2, 4]
# y = line_by_point_and_slope(point, 2, x)
# y = line_by_point_and_vector(point, [2, 1], x)
# y = line_by_two_points(point, point2, x)
# y = line_by_point_and_normal_vector(point, [3, -1], x)

# plt.figure()
# plt.plot(x, y)
# plt.scatter(point[0], point[1])
# plt.quiver([-2], [-1], [3], [-1], angles='xy', scale_units='xy', scale=1)
# plt.quiver([2], [1], [1], [3], angles='xy', scale_units='xy', scale=1)
# # plt.scatter(point2[0], point2[1])
# plt.xlim(-3, 3)
# plt.ylim(-3, 5)
# plt.grid()
# plt.show()

# v = np.array([[1, 1], [3, 5], [-5, -5]])

# rows, cols = v.T

# plt.figure()
# plt.quiver([0, 0, 0], [0, 0, 0], rows, cols, angles='xy', scale_units='xy', scale=1)
# plt.xlim(-10, 10)
# plt.ylim(-10, 10)
# plt.grid()
# plt.show()
