import numpy as np


def derivative_newtons_method(f, x, dx=0.01): return (f(x+dx) - f(x)) / dx


def derivative(f, x, dx=0.01): return (f(x + dx) - f(x - dx))/(2 * dx)


def derivative_function(f, dx=0.01): return lambda x: derivative(f, x, dx)


def derivative_by_points(points, dx):
    derivative = []
    for i in range(1, len(points)-1):
        result = (points[i+1] - points[i - 1]) / (2 * dx)
        derivative.append(result)
    return np.array(derivative)


def higher_order_derivative(points, dx, order):
    derivative = points
    for i in range(0, order):
        derivative = derivative_by_points(derivative, dx)
    return derivative


def tangent_line_of_curve(f, a, dx=0.01):
    return lambda x: f(a) + derivative(f, a, dx) * (x - a)


def normal_line_of_curve(f, a, dx=0.01):
    df = derivative(f, a, dx)
    if abs(df) == 0:
        return lambda x: f(a) * x
    return lambda x: f(a) + (-1 * np.power(derivative(f, a, dx), -1)) * (x - a)


def factorial(x):
    result = 1
    for i in range(1, x+1):
        result *= i
    return result


def limit(f, x, e=0.001):
    return (f(x-e) + f(x + e)) / 2


def left_point_rule(f, a, b, n):
    h = (b - a) / n
    x = np.arange(a, b, h)
    y = f(x)
    s = y.sum() * h
    return s


def right_point_rule(f, a, b, n):
    h = (b - a) / n
    x = np.arange(a+h, b+h, h)
    y = f(x)
    s = y.sum() * h
    return s


def mid_point_rule(f, a, b, n):
    h = (b - a) / n
    x = np.arange(a+(h/2), b+(h/2), h)
    y = f(x)
    s = y.sum() * h
    return s


def trapezoidal_rule(f, a, b, n):
    h = (b - a) / n
    if h == 0:
        return 0
    x = np.arange(a, b + h, h)
    y = f(x)
    s = h * ((y[0] + y[-1])/2 + y[1:-1].sum())
    return s


def simpsons_rule(f, a, b, n=10):
    h = (b - a) / (2 * n)
    if h == 0:
        return 0
    x = np.arange(a, b + h, h)
    y = f(x)
    even = y[::2]
    odd = y[1::2]
    s = (h / 3) * (y[0] + y[-1] + 2*even[1:-1].sum() + 4*odd.sum())
    return s


def arc_length(f, a, b, n):
    dx = (b - a) / (2 * n)
    l = simpsons_rule(lambda x: np.sqrt(1 + (derivative(f, x, dx)**2)), a, b, n)
    return l


def average_value(f, a, b, n):
    value = 1/(b-a) * simpsons_rule(f, a, b, n)
    return value


def surface_area_of_revolution_about_y(f, a, b, n):
    dx = (b - a) / (2 * n)
    area = simpsons_rule(lambda x: 2 * x * np.pi * np.sqrt(1 + (derivative(f, x, dx)**2)), a, b, n)
    return area


def surface_area_of_revolution_about_x(f, a, b, n):
    dx = (b - a) / (2 * n)
    area = simpsons_rule(lambda x: 2 * f(x) * np.pi * np.sqrt(1 + (derivative(f, x, dx)**2)), a, b, n)
    return area


def volume_of_revolution(f, a, b, n):
    volume = simpsons_rule(lambda x: np.pi * np.power(f(x), 2), a, b, n)
    return volume


def volume_of_revolution_between_curves(f1, f2, a, b, n):
    volume = simpsons_rule(lambda x: np.pi * (np.power(f1(x), 2) - np.power(f2(x), 2)), a, b, n)
    return volume


def moments_of_system(f1, f2, a, b, n):
    Mx = simpsons_rule(lambda x: 0.5 * (np.power(f1(x), 2) - np.power(f2(x), 2)), a, b, n)
    My = simpsons_rule(lambda x: x * (f1(x) - f2(x)), a, b, n)
    return (Mx, My)


def center_of_mass(f1, a, b, n):
    area = simpsons_rule(lambda x: f1(x), a, b, n)
    Mx, My = moments_of_system(f1, lambda x: 0, a, b, n)
    x = My / area
    y = Mx / area
    return (x, y)


def center_of_mass_between_curves(f1, f2, a, b, n):
    area = simpsons_rule(lambda x: f1(x) - f2(x), a, b, n)
    Mx, My = moments_of_system(f1, f2, a, b, n)
    x = My / area
    y = Mx / area
    return (x, y)


def series_ratio_test(f, limval=99999):
    return limit(lambda x: np.abs(f(x+1) * np.power(f(x), -1)), limval)


def series_root_test(f, limval=99999):
    return limit(lambda x: np.power(np.abs(f(x+1) * np.power(f(x), -1)), 1/x), limval)


def series_integral_test(f, a=1, limval=99999):
    return simpsons_rule(f, a+1, limval + 1, 100) / simpsons_rule(f, a, limval, 100)


def line_by_point_and_slope(point, slope):
    px, py = point
    return lambda x: py + slope*(x - px)


def line_by_point_and_vector(point, vector):
    px, py = point
    vx, vy = vector
    if vx == 0 or vy == 0:
        return lambda x: 0
    return lambda x: py + (vy * ((x - px) / vx))


def line_by_two_points(p1, p2, x):
    vector = [p2[0] - p1[0], p2[1] - p1[1]]
    return line_by_point_and_vector(p1, vector, x)


def line_by_point_and_normal_vector(point, normal):
    px, py = point
    nx, ny = normal
    if nx == 0 or ny == 0:
        return lambda x: 0
    return lambda x: py - ((nx * (x - px)) / ny)


def distance_between_point_and_line(point, a, b, c):
    px, py = point
    num = abs(a*px + b*py + c)
    den = np.sqrt(a*a + b*b)
    return num/den


def distance_between_point_and_plane(point, a, b, c, d):
    px, py, pz = point
    num = abs(a*px + b*py + c*pz + d)
    den = np.sqrt(a*a + b * b + c*c)
    return num/den


def distance_between_parallel_planes(a, b, c, d1, d2):
    return abs(d2-d1) / np.sqrt(a*a + b*b + c*c)


def angle_between_planes(p1, p2):
    a1, b1, _ = p1
    a2, b2, _ = p2
    tg = (a1*b2 - b1*a2) / (a1*a2 + b1*b2)
    return np.arctan(tg)


def angle_between_vectors_3D(v1, v2):
    return np.arccos(vector_dot_3D(v1, v2) / (vector_magnitude_3D(v1) * vector_magnitude_3D(v2)))


def angle_between_vectors(v1, v2):
    v1x, v1y = v1
    v2x, v2y = v2
    return angle_between_vectors_3D([v1x, v1y, 0], [v2x, v2y, 0])


def vector_dot_3D(v1, v2):
    v1x, v1y, v1z = v1
    v2x, v2y, v2z = v2
    return v1x*v2x+v1y*v2y+v1z*v2z


def vector_dot(v1, v2):
    v1x, v1y = v1
    v2x, v2y = v2
    return vector_dot_3D([v1x, v1y, 0], [v2x, v2y, 0])


def vector_cross_3D(v1, v2):
    v1x, v1y, v1z = v1
    v2x, v2y, v2z = v2
    i = (v1y*v2z) - (v2y*v1z)
    j = -((v1x*v2z) - (v2x*v1z))
    k = (v1x*v2y) - (v2x*v1y)
    return np.array([i, j, k])


def vector_cross(v1, v2):
    v1x, v1y = v1
    v2x, v2y = v2
    return vector_cross_3D([v1x, v1y, 0], [v2x, v2y, 0])


def vector_magnitude_3D(v):
    vx, vy, vz = v
    return np.sqrt(vx**2+vy**2+vz**2)


def vector_magnitude(v):
    vx, vy = v
    return vector_magnitude_3D([vx, vy, 0])


def projection_on_vector(v, p):
    return projection_on_vector_3D([v[0], v[1], 0], [p[0], p[1], 0])


def projection_on_vector_3D(v, p):
    vx, vy, vz = v
    px, py, pz = p
    return (vx*px + vy*py + vz*pz) / np.sqrt(px**2 + py**2 + pz**2)


def fourier_series(f, l, n, x):
    a0 = 1/l * simpsons_rule(f, -l, l, 10)
    a1 = 1/l * simpsons_rule(lambda x: f(x)*np.cos(x), -l, l, 10)
    b1 = 1/l * simpsons_rule(lambda x: f(x)*np.sin(x), -l, l, 10)
    result = a0/2 + a1*np.cos(x) + b1*np.sin(x)
    for i in range(2, n+1):
        ai = 1/l * simpsons_rule(lambda x: f(x)*np.cos(i*x * (np.pi / l)), -l, l, 10)
        bi = 1/l * simpsons_rule(lambda x: f(x)*np.sin(i*x * (np.pi / l)), -l, l, 10)
        result += ai*np.cos(i*x) + bi*np.sin(i*x)
    return result


def direction_cosines(vector):
    return direction_cosines_3D([vector[0], vector[1], 0])[:-1]


def direction_cosines_3D(vector):
    x, y, z = vector
    magnitude = np.sqrt(x**2 + y**2 + z**2)
    cosa = x/magnitude
    cosb = y/magnitude
    cosy = z/magnitude
    return np.array([cosa, cosb, cosy])


def partial_derivative_2D_by_x(f, point, dx=0.01):
    x, y = point
    return (f(x+dx, y) - f(x-dx, y)) / (2*dx)


def partial_derivative_2D_by_y(f, point, dy=0.01):
    x, y = point
    return (f(x, y + dy) - f(x, y - dy)) / (2*dy)


def partial_derivative_3D_by_x(f, point, dx=0.01):
    x, y, z = point
    return (f(x+dx, y, z) - f(x-dx, y, z)) / (2*dx)


def partial_derivative_3D_by_y(f, point, dy=0.01):
    x, y, z = point
    return (f(x, y + dy, z) - f(x, y - dy, z)) / (2*dy)


def partial_derivative_3D_by_z(f, point, dz=0.01):
    x, y, z = point
    return (f(x, y, z + dz) - f(x, y, z - dz)) / (2*dz)


def directional_derivative(f, point, vector):
    return directional_derivative_3D(lambda x, y, z: f(x, y), [point[0], point[1], 0], [vector[0], vector[1], 0])


def directional_derivative_3D(f, point, vector):
    dfdx = partial_derivative_3D_by_x(f, point)
    dfdy = partial_derivative_3D_by_y(f, point)
    dfdz = partial_derivative_3D_by_z(f, point)
    cosa, cosb, cosy = direction_cosines_3D(vector)
    return dfdx*cosa + dfdy*cosb + dfdz*cosy


def gradient_vector(f, point):
    return gradient_vector_3D(lambda x, y, z: f(x, y), [point[0], point[1], 0])[:-1]


def gradient_vector_3D(f, point):
    dfdx = partial_derivative_3D_by_x(f, point)
    dfdy = partial_derivative_3D_by_y(f, point)
    dfdz = partial_derivative_3D_by_z(f, point)
    return np.array([dfdx, dfdy, dfdz])


def partial_derivatives_2D_by_points(points, dx):
    dfdx = np.array([])
    dfdy = np.array([])

    for i in range(points.shape[0]):
        dfdxi = derivative_by_points(points[i], dx)
        dfdx = np.append(dfdx, dfdxi)
    dfdx = dfdx.reshape(points.shape[0], -1)

    for j in range(points.shape[1]):
        dfdyj = derivative_by_points(points[:, j], dx)
        dfdy = np.append(dfdy, dfdyj)
    dfdy = dfdy.reshape(points.shape[1], -1).T
    return [dfdx, dfdy]


def least_squares(x, y):
    x_sum = x.sum()
    y_sum = y.sum()
    x2 = x**2
    x2_sum = x2.sum()
    xy = x*y
    xy_sum = xy.sum()
    n = len(x)
    det = x2_sum * n - x_sum*x_sum
    det_a = xy_sum * n - y_sum*x_sum
    det_b = x2_sum * y_sum - x_sum * xy_sum
    a = det_a / det
    b = det_b / det
    return lambda x: a*x + b


def tangent_plane(f, point):
    px, py, pz = point
    dfdx = partial_derivative_3D_by_x(f, point)
    dfdy = partial_derivative_3D_by_y(f, point)
    return lambda x, y: dfdx[0]*(x - px) + dfdy[0]*(y - py) + pz


def normal_plane(f, point):
    px, py, pz = point
    dfdx = partial_derivative_3D_by_x(f, point)
    dfdy = partial_derivative_3D_by_y(f, point)
    return lambda x, y: 1/dfdx[0]*(x - px) + 1/dfdy[0]*(y - py) + pz


def normal_vector_3D(f, point):
    return gradient_vector_3D(f, point)


def normal_line_2D(f):
    dfdx = partial_derivative_2D_by_x(f)


def vector_function(fi, fj, fk):
    return lambda t: np.array([fi(t), fj(t), fk(t)], dtype=object)


def derivative_of_vector_function(vectors, dx):
    i = derivative_by_points(vectors[0], dx)
    j = derivative_by_points(vectors[1], dx)
    k = derivative_by_points(vectors[2], dx)
    return np.array([i, j, k])


def derivative_of_vector_function_at_point(f, point, dx=0.01):
    t = np.linspace(point - dx, point + dx, 3)
    vectors = f(t)
    return derivative_of_vector_function(vectors, dx)


def unit_tangent_vector(fi, fj, fk, point):
    di = derivative(fi, point)
    dj = derivative(fj, point)
    dk = derivative(fk, point)
    magnitude = vector_magnitude_3D([di, dj, dk])
    return [di / magnitude, dj / magnitude, dk / magnitude]


def unit_normal_vector(fi, fj, fk, point):
    f = vector_function(fi, fj, fk)
    t = np.linspace(point - 0.02, point + 0.02, 5)
    vectors = f(t)
    dfdt = derivative_of_vector_function(vectors, 0.01)
    didt, djdt, dkdt = dfdt
    magnitude_f = vector_magnitude_3D([didt[1], djdt[1], dkdt[1]])
    dfdt = dfdt / magnitude_f
    d2fdt = derivative_of_vector_function(dfdt, 0.01)
    d2idt, d2jdt, d2kdt = d2fdt
    magnitude_t = vector_magnitude_3D([d2idt[0], d2jdt[0], d2kdt[0]])
    return [d2idt[0] / magnitude_t, d2jdt[0] / magnitude_t, d2kdt[0] / magnitude_t]


def binormal_vector(fi, fj, fk, point):
    tangent = unit_tangent_vector(fi, fj, fk, point)
    normal = unit_normal_vector(fi, fj, fk, point)
    return vector_cross_3D(tangent, normal)


def tangential_comp_of_acceleration(fv, point, dx=0.01):
    t = np.linspace(point - (2*dx), point + (2*dx), 5)
    vectors = fv(t)
    dfdt = derivative_of_vector_function(vectors, dx)
    d2fdt = derivative_of_vector_function(dfdt, dx)
    return vector_dot_3D([dfdt[0][1], dfdt[1][1], dfdt[2][1]], [d2fdt[0][0], d2fdt[1][0], d2fdt[2][0]]) / vector_magnitude_3D([dfdt[0][1], dfdt[1][1], dfdt[2][1]])


def normal_comp_of_acceleration(fv, point, dx=0.01):
    t = np.linspace(point - (2*dx), point + (2*dx), 5)
    vectors = fv(t)
    dfdt = derivative_of_vector_function(vectors, dx)
    d2fdt = derivative_of_vector_function(dfdt, dx)
    return vector_magnitude_3D(vector_cross_3D([dfdt[0][1], dfdt[1][1], dfdt[2][1]], [d2fdt[0][0], d2fdt[1][0], d2fdt[2][0]])) / vector_magnitude_3D([dfdt[0][1], dfdt[1][1], dfdt[2][1]])


def arc_length_of_vector_function(fi, fj, fk, a, b, n=10):
    return simpsons_rule(lambda t: np.sqrt(derivative(fi, t)**2 + derivative(fj, t)**2 + derivative(fk, t)**2), a, b, n)


# def average_value2D(f, range_x, range_y, xn, yn):
#     start_x, end_x = range_x
#     start_y, end_y = range_y
#     delta_x = (end_x - start_x) / xn
#     delta_y = (end_y - start_y) / yn
#     delta_area = delta_x * delta_y
#     area = end_x * end_y
#     x = np.linspace(start_x + (delta_x/2), end_x - (delta_x/2), xn)
#     y = np.linspace(start_y + (delta_y/2), end_y - (delta_y/2), yn)
#     x, y = np.meshgrid(x, y)
#     z = f(x, y)
#     z_sum = 0
#     for i in range(len(z)):
#         z_sum += z[i].sum()
#     return (1 / area) * (delta_area * z_sum)


def convert_to_polar_function(f):
    return lambda r, phi: f(r*np.cos(phi), r*np.sin(phi))


def convert_to_cylindrical_function(f):
    return lambda r, phi, z: f(r*np.cos(phi), r*np.sin(phi), z)


def convert_to_spherical_function(f):
    return lambda r, t, phi: f(r*np.sin(t)*np.cos(phi), r*np.sin(t)*np.sin(phi), r*np.cos(t))


def swap_functions_parameters_2D(f):
    return lambda x, y: f(y, x)


def swap_functions_parameters_3D(f, axis):
    if axis == "xzy":
        return lambda x, y, z: f(x, z, y)
    if axis == "yxz":
        return lambda x, y, z: f(y, x, z)
    if axis == "yzx":
        return lambda x, y, z: f(y, z, x)
    if axis == "zxy":
        return lambda x, y, z: f(z, x, y)
    if axis == "zyx":
        return lambda x, y, z: f(z, y, x)


def double_integral(f, range, f_lower, f_upper, axis=0, xn=10, yn=10):
    if axis == 1:
        f = swap_functions_parameters_2D(f)
    start_x, end_x = range
    delta_x = (end_x - start_x) / xn
    x = np.linspace(start_x + (delta_x/2), end_x - (delta_x/2), xn)
    z_sum = 0
    # print(f(1, 1))
    # print(f(0, 0))
    # print(f(1, 0))
    # print(f(2, 0))
    # print(f(0, -1))
    # print(f(0, 0))
    # print(f(0, 1))
    # print(f(0, 2))
    for i in x:
        start_y = f_lower(i)
        end_y = f_upper(i)
        delta_y = (end_y - start_y) / yn
        delta_area = delta_x * delta_y
        y = np.linspace(start_y + (delta_y/2), end_y - (delta_y/2), yn)
        f_sum = 0
        for j in y:
            f_sum += f(i, j)
        f_sum *= delta_area
        z_sum += f_sum
    return z_sum


def triple_integral(f, range_x, fy_lower, fy_upper, fz_lower, fz_upper, axis="xyz", xn=10, yn=10, zn=10):
    if axis != "xyz":
        f = swap_functions_parameters_3D(f, axis)
    start_x, end_x = range_x
    delta_x = (end_x - start_x) / xn
    x = np.linspace(start_x + (delta_x/2), end_x - (delta_x/2), xn)
    result = 0
    for i in x:
        start_y = fy_lower(i)
        end_y = fy_upper(i)
        delta_y = (end_y - start_y) / yn
        y = np.linspace(start_y + (delta_y/2), end_y - (delta_y/2), yn)
        f_sum = 0
        for j in y:
            start_z = fz_lower(i, j)
            end_z = fz_upper(i, j)
            delta_z = (end_z - start_z) / zn
            delta_area = delta_x * delta_y * delta_z
            z = np.linspace(start_z + (delta_z/2), end_z - (delta_z/2), zn)
            for k in z:
                f_sum += f(i, j, k)
            f_sum *= delta_area
            result += f_sum
    return result


def average_value_2D(f, range, f_lower, f_upper, axis=0, xn=10, yn=10):
    return double_integral(f, range, f_lower, f_upper, axis, xn, yn) / double_integral(lambda x, y: 1, range, f_lower, f_upper, axis, xn, yn)


def surface_area_2D(f, range, f_lower, f_upper, axis=0, xn=10, yn=10):
    return double_integral(lambda x, y: np.sqrt(1 + partial_derivative_2D_by_x(f, [x, y])**2 +
                                                partial_derivative_2D_by_y(f, [x, y])**2), range, f_lower, f_upper, axis, xn, yn)


def center_of_mass_2D(f, range, f_lower, f_upper, axis=0, xn=10, yn=10):
    m = double_integral(f, range, f_lower, f_upper, axis, xn, yn)
    lx = double_integral(lambda x, y: x * f(x, y), range, f_lower, f_upper, axis, xn, yn)
    ly = double_integral(lambda x, y: y * f(x, y), range, f_lower, f_upper, axis, xn, yn)
    x = lx / m
    y = ly / m
    return [x, y]


def center_of_mass_3D(f, range_x, fy_lower, fy_upper, fz_lower, fz_upper, xn=10, yn=10, zn=10):
    m = triple_integral(f, range_x, fy_lower, fy_upper, fz_lower, fz_upper, xn, yn, zn)
    lx = triple_integral(lambda x, y, z: x * f(x, y, z), range_x, fy_lower, fy_upper, fz_lower, fz_upper, xn, yn, zn)
    ly = triple_integral(lambda x, y, z: y * f(x, y, z), range_x, fy_lower, fy_upper, fz_lower, fz_upper, xn, yn, zn)
    lz = triple_integral(lambda x, y, z: z * f(x, y, z), range_x, fy_lower, fy_upper, fz_lower, fz_upper, xn, yn, zn)
    x = lx / m
    y = ly / m
    z = lz / m
    return [x, y, z]


def curvature(points):
    dfdx = derivative_by_points(points, 0.01)
    d2fdx = derivative_by_points(dfdx, 0.01)
    dfdx = dfdx[1:-1]
    return d2fdx / np.power(np.sqrt(1 + dfdx*dfdx), 3)


def line_integral(f, a, b, line):
    return simpsons_rule(lambda x: f(x, line(x)) * np.sqrt(1 + derivative(line, x)**2), a, b)


def line_integral_3D(f, a, b, xt, yt, zt):
    return simpsons_rule(lambda t: f(xt(t), yt(t), zt(t)) *
                         np.sqrt(derivative(xt, t)**2 + derivative(yt, t)**2 + derivative(zt, t)**2), a, b)


def line_integral_by_parametric_curve(f, a, b, xt, yt):
    return simpsons_rule(lambda t: f(xt(t), yt(t)) * np.sqrt(derivative(xt, t)**2 + derivative(yt, t)**2), a, b)


def line_integral_of_2form_greens_rule(p, q, range, f_lower, f_upper, axis=0, xn=10, yn=10):
    return double_integral(lambda x, y: partial_derivative_2D_by_x(q, [x, y]) - partial_derivative_2D_by_y(p, [x, y]),
                           range, f_lower, f_upper, axis, xn, yn)


def surface_integral(f, surface, range, f_lower, f_upper, proj="xy", axis=0, xn=10, yn=10):
    result = 0
    if proj == "yz":
        result = double_integral(lambda y, z: f(surface(y, z), y, z) *
                                 np.sqrt(partial_derivative_2D_by_x(surface, [y, z])**2 +
                                         partial_derivative_2D_by_y(surface, [y, z])**2 + 1),
                                 range, f_lower, f_upper, axis, xn, yn)
    if proj == "xz":
        result = double_integral(lambda x, z: f(x, surface(x, z), z) *
                                 np.sqrt(partial_derivative_2D_by_x(surface, [x, z])**2 +
                                         partial_derivative_2D_by_y(surface, [x, z])**2 + 1),
                                 range, f_lower, f_upper, axis, xn, yn)
    if proj == "xy":
        result = double_integral(lambda x, y: f(x, y, surface(x, y)) *
                                 np.sqrt(partial_derivative_2D_by_x(surface, [x, y])**2 +
                                         partial_derivative_2D_by_y(surface, [x, y])**2 + 1),
                                 range, f_lower, f_upper, axis, xn, yn)

    # print(f(0, -1, 0))
    # print(f(1, -1, 0))
    # print(f(0, 0, 0))
    # print(f(0, 1, 0))
    # print(f(0, 2, 0))
    return result


def surface_integral_of_2form(p, q, r, surface, range, f_lower, f_upper, proj="xy", axis=0, xn=10, yn=10):
    # dsdx = partial_derivative_2D_by_x(surface, [-1.94, -0.40])
    # dsdy = partial_derivative_2D_by_y(surface, [-1.94, -0.40])
    # denominator = np.sqrt(dsdx**2 + dsdy**2 + 1)
    # i = -dsdx / denominator
    # j = -dsdy / denominator
    # k = 1 / denominator
    return surface_integral(lambda x, y, z:
                            p(x, y, z)*(-partial_derivative_2D_by_x(surface, [x, y]) /
                                        np.sqrt(partial_derivative_2D_by_x(surface, [x, y])**2 + partial_derivative_2D_by_y(surface, [x, y])**2 + 1)) +
                            q(x, y, z)*(-partial_derivative_2D_by_y(surface, [x, y]) /
                                        np.sqrt(partial_derivative_2D_by_x(surface, [x, y])**2 + partial_derivative_2D_by_y(surface, [x, y])**2 + 1)) +
                            r(x, y, z)*(1 /
                                        np.sqrt(partial_derivative_2D_by_x(surface, [x, y])**2 + partial_derivative_2D_by_y(surface, [x, y])**2 + 1)),
                            surface, range, f_lower, f_upper, proj, axis, xn, yn)
    # return surface_integral(lambda x, y, z:
    #                         p(x, y, z)*(partial_derivative_2D_by_x(surface, [y, z]) /
    #                                     np.sqrt(partial_derivative_2D_by_x(surface, [y, z])**2 + partial_derivative_2D_by_y(surface, [y, z])**2 + 1)) +
    #                         q(x, y, z)*(partial_derivative_2D_by_y(surface, [y, z]) /
    #                                     np.sqrt(partial_derivative_2D_by_x(surface, [y, z])**2 + partial_derivative_2D_by_y(surface, [y, z])**2 + 1)) +
    #                         r(x, y, z)*(-1 /
    #                                     np.sqrt(partial_derivative_2D_by_x(surface, [y, z])**2 + partial_derivative_2D_by_y(surface, [y, z])**2 + 1)),
    #                         surface, range, f_lower, f_upper, proj, axis, xn, yn)


def surface_integral_greens_rule(p, q, r, range, f1_lower, f1_upper, f2_lower, f2_upper, axis="xyz", xn=10, yn=10, zn=10):
    return triple_integral(lambda x, y, z: partial_derivative_3D_by_x(p, [x, y, z]) + partial_derivative_3D_by_y(q, [x, y, z]) +
                           partial_derivative_3D_by_z(r, [x, y, z]), range, f1_lower, f1_upper, f2_lower, f2_upper, axis, xn, yn, zn)


def simple_euler_method(f, a, b, y0, n):
    h = (b - a) / n
    x = np.arange(a, b+h, h)
    y = [y0]
    for i in range(1, len(x)):
        yi = y[i - 1] + h*f(x[i - 1], y[i-1])
        y.append(yi)
    return [x, np.array(y)]


def improved_euler_method(f, a, b, y0, n):
    h = (b - a) / n
    x = np.arange(a, b+h, h)
    y = [y0]
    for i in range(1, len(x)):
        yi = y[i - 1] + h*f(x[i - 1] + (h/2), y[i-1]+(h/2)*f(x[i - 1], y[i-1]))
        y.append(yi)
    return [x, np.array(y)]


def runge_kutta_method(f, a, b, y0, n):
    h = (b - a) / n
    x = np.arange(a, b+h, h)
    y = [y0]
    for i in range(1, len(x)):
        k1 = f(x[i-1], y[i-1])
        k2 = f(x[i-1] + (h/2), y[i-1] + (h*k1*0.5))
        k3 = f(x[i-1] + (h/2), y[i-1] + (h*k2*0.5))
        k4 = f(x[i-1] + h, y[i-1] + (h*k3))
        yi = y[i - 1] + (h/6)*(k1+2*k2+2*k3+k4)
        y.append(yi)
    return [x, np.array(y)]
