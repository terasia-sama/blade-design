#coding:utf-8
#几何构造、计算实用包

import numpy as np
import sys
import sympy
import copy


def angle_between_vec(vec1, vec2):
    """返回平面上两个二维矢量的夹角，以x轴正方向为0度，逆时针旋转，[0,pi]"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    if vec1.shape[0] != 2:
        print("输入的第一个向量不是一个二维向量")
        sys.exit()
    elif vec2.shape[0] != 2:
        print("输入的第二个向量不是一个二维向量")
        sys.exit()
    angle = np.arccos(np.dot(vec1, vec2) / np.linalg.norm(vec1) / np.linalg.norm(vec2))
    return angle


def angle_of_vec(vec):
    """返回一个二维矢量的平面夹角，以x轴正方向为0度，逆时针旋转，[0,2*pi)"""
    vec = np.array(vec)
    if vec.shape[0] != 2:
        print("该输入不是一个二维向量")
        sys.exit()

    if vec[0] == 0.0:
        if vec[1] > 0:
            angle = np.pi / 2
        else:
            angle = 3 * np.pi / 2
    elif vec[0] < 0.0:
        angle = np.arctan(vec[1] / vec[0]) + np.pi
    elif vec[1] < 0.0:
        angle = np.arctan(vec[1] / vec[0]) + 2.0 * np.pi
    else:
        angle = np.arctan(vec[1] / vec[0])
    return angle


def rotate_points(pts, alpha, pt0):
    """将点（集）绕某点pt0逆时针旋转alpha (rad)角度  return: 点（集）"""
    npt = pts.shape[0]
    if pts.ndim == 1:
        new_pt = np.zeros(2,dtype=float, order='C')
        new_pt[0] = (pts[0] - pt0[0]) * np.cos(alpha) - (pts[1] - pt0[1]) * np.sin(alpha) + pt0[0]
        new_pt[1] = (pts[0] - pt0[0]) * np.sin(alpha) + (pts[1] - pt0[1]) * np.cos(alpha) + pt0[1]
        return new_pt
    else:
        new_pt = np.zeros([npt, 2], dtype=float, order='C')
        for i in range(npt):
            new_pt[i, 0] = (pts[i, 0] - pt0[0]) * np.cos(alpha) - (pts[i, 1] - pt0[1]) * np.sin(alpha) + pt0[0]
            new_pt[i, 1] = (pts[i, 0] - pt0[0]) * np.sin(alpha) + (pts[i, 1] - pt0[1]) * np.cos(alpha) + pt0[1]
        return new_pt


def normalization(x):
    vec = np.array(x)
    if vec.ndim == 2:
        y = np.zeros_like(vec)
        for i in range(vec.shape[0]):
            y[i, :] = vec[i, :] / np.linalg.norm(vec[i, :])
    elif vec.ndim == 1:
        y = vec / np.linalg.norm(vec)
    else:
        print('输入数组维度超过2维，尚无对其长度归一化的方法')
        sys.exit()
    return y


def solve_eqs_2x2th(coef0, coef1):
    x = sympy.symbols('x')
    y = sympy.symbols('y')
    solution = sympy.solve([coef0[0] * x ** 2 + coef0[1] * y + coef0[2], coef1[0] * y ** 2 + coef1[1] * x + coef1[2]],
                           [x, y])
    solution = np.array(solution)
    n = solution.shape[0]
    print(solution)
    s0 = np.zeros([2, 2], dtype=float, order='C')
    k = 0
    for i in range(n):
        solution[i, 0] = complex(solution[i, 0])
        solution[i, 1] = complex(solution[i, 1])
        if solution[i, 0].imag == 0 and solution[i, 1].imag == 0:
            s0[k, 0] = solution[i, 0].real
            s0[k, 1] = solution[i, 1].real
            k += 1
    return solution


def solve_proportion(k0, k3, v0, v3, a):
    coef = np.zeros([5], dtype=float, order='C')
    coef[4] = 1.5 ** 3 * k0 * k3 ** 2 / (np.cross(v3, v0)) ** 2
    coef[3] = 0.0
    coef[2] = 4.5 * k0 * k3 * np.cross(v3, a) / (np.cross(v3, v0)) ** 2
    coef[1] = - np.cross(v0, v3)
    coef[0] = 1.5 * k0 * (np.cross(v3, a)) ** 2 / (np.cross(v3, v0)) ** 2 - np.cross(v0, a)
    y0 = sympy.symbols('y0')
    # kappa_3
    result = sympy.solve([coef[4] * y0 ** 4 + coef[2] * y0 ** 2 + coef[1] * y0 + coef[0]], [y0])
    print(result)

    sy = np.zeros([4], dtype=float, order='C')
    k = 0
    for i in range(len(result)):
        temp = complex(result[i][0])
        if abs(temp.imag) < 10 ** (-9):
            if temp.real < 0:
                sy[k] = temp.real
                k += 1

    x = y = 0
    for i in range(k):
        x = (1.5 * k3 * sy[i] ** 2 + np.cross(v3, a)) / np.cross(v3, v0)
        # print(x,sy[i])
        if x > 0:
            y = sy[i]
    print('x, y = ', x, y)
    return np.array([x, y])


def solve_costs(d, ds, dp, m, beta_s, beta_p):
    coef = np.zeros([3], dtype=float, order='C')
    coef[0] = d * np.tan(beta_s) * np.tan(beta_p)
    coef[1] = ds * np.tan(beta_p) - dp * m * np.tan(beta_s) - d * (m + 1) * np.tan(beta_s) * np.tan(beta_p)
    coef[2] = d * m * np.tan(beta_s) * np.tan(beta_p) + dp * m * np.tan(beta_s) - ds * m * np.tan(beta_p)
    x = sympy.symbols('x')
    result = sympy.solve([coef[2] * x ** 2 + coef[1] * x + coef[0]], [x])
    print(result)

    costs = 0
    for i in range(len(result)):
        temp = complex(result[i][0])
        if abs(temp.imag) < 10 ** (-10) and -1 < temp.real < 0 and -1 < m * temp.real < 0:
            costs = temp.real
            break
    # costs = complex(result[1][0]).real
    print('costs', costs)
    return costs


def find_seg_intersection(line_seg1_a, line_seg1_b, line_seg2_c, line_seg2_d):
    a = line_seg1_a
    b = line_seg1_b
    c = line_seg2_c
    d = line_seg2_d
    # 快速排斥试验筛除
    intersection = 0
    if min(a[0], b[0]) < max(c[0], d[0]) and min(c[0], d[0]) < max(a[0], b[0]):
        if min(a[1], b[1]) < max(c[1], d[1]) and min(c[1], d[1]) < max(a[1], b[1]):
            # print('快速排斥试验筛除 pass')
            # 跨立试验
            if np.cross(b - a, c - a) * np.cross(b - a, d - a) <= 0:
                if np.cross(d - c, a - c) * np.cross(d - c, b - c) <= 0:
                    # 求交点坐标
                    # print('跨立试验 pass')
                    if np.cross(b - a, d - c) == 0:
                        print('np.cross(b - a, d - c) = 0')
                    lambda1 = np.cross(c - a, d - c) / np.cross(b - a, d - c)
                    # print('lambda1 = ', lambda1)
                    if 0 <= lambda1 <= 1:
                        intersection = a + lambda1 * (b - a)
                        # intersection = np.array(intersection)
                        # print('intersection = ', intersection)
                    else:
                        print('lambda1 = ', lambda1, ', beyond [0,1]')
                        sys.exit()
    return intersection


def modify_endpoint(curv1, curv2, mode):
    tan_vec1 = np.array([curv1[0, 0] - curv1[1, 0], curv1[0, 1] - curv1[1, 1]])
    tan_vec1 = tan_vec1 / np.linalg.norm(tan_vec1)
    tan_vec2 = np.array([curv2[0, 0] - curv2[1, 0], curv2[0, 1] - curv2[1, 1]])
    tan_vec2 = tan_vec2 / np.linalg.norm(tan_vec2)
    alpha1 = angle_between_vec(tan_vec1, curv2[0, :] - curv1[0, :])
    alpha2 = angle_between_vec(tan_vec2, curv1[0, :] - curv2[0, :])
    kappa = np.linalg.norm(curv1[0, :] - curv2[0, :]) * np.sin(abs(alpha1 - alpha2) / 2) / np.sin(
        (alpha1 + alpha2) / 2)
    if mode == 'add' and alpha1 > alpha2:
        curv2 = np.insert(curv2, 0, curv2[0, :] + kappa * tan_vec2, axis=0)
    elif mode == 'add' and alpha1 < alpha2:
        curv1 = np.insert(curv1, 0, curv1[0, :] + kappa * tan_vec1, axis=0)
    elif mode == 'cut' and alpha1 > alpha2:
        curv1[0, :] = curv1[0, :] - kappa * tan_vec1
    elif mode == 'cut' and alpha1 < alpha2:
        curv2[0, :] = curv2[0, :] - kappa * tan_vec2
    return curv1, curv2


def calc_line_intersection(k1, pt1, k2, pt2):
    """斜率1，点1，斜率2，点2"""
    if k1 != k2:
        x = k1 * pt1[0] - k2 * pt2[0] + pt2[1] - pt1[1]
        y = k1 * (x - pt1[0]) + pt1[1]
        return np.array([x, y])
    else:
        print('输入两直线平行，无法计算唯一交点！')
        sys.exit()


def calc_tan_vec_by_difference(pt):
    pt = np.array(pt)
    npt = pt.shape[0]
    ndim = pt.shape[1]

    tan_vec = np.zeros([npt, ndim], dtype=float, order='C')
    tan_vec[0, :] = pt[1, :] - pt[0, :]
    tan_vec[-1, :] = pt[-1, :] - pt[-2, :]
    for i in range(1, npt-1):
        tan_vec[i, :] = pt[i+1, :] - pt[i-1, :]
    return normalization(tan_vec)


def calc_curva_by_difference(pt):
    pt = np.array(pt)

    tan_vec = calc_tan_vec_by_difference(pt)
    dy = copy.deepcopy(pt)
    dy[:, 1] = tan_vec[:, 1] / tan_vec[:, 0]
    tan_vec2 = calc_tan_vec_by_difference(dy)
    d2y = tan_vec2[:, 1] / tan_vec2[:, 0]

    curva = d2y / (1 + dy[:, 1]**2)**1.5
    return np.array(curva)


# 从leading_edge移入
def return_begin_vec(curv):
    temp = curv[1, :] - curv[0, :]
    return temp / np.linalg.norm(temp)


def return_begin_curva(curv):
    dy0 = (curv[1, 1] - curv[0, 1]) / (curv[1, 0] - curv[0, 0])
    dy1 = (curv[2, 1] - curv[0, 1]) / (curv[2, 0] - curv[0, 0])
    d2y0 = (dy1 - dy0) / (curv[1, 0] - curv[0, 0])
    return d2y0 / (1. + dy0**2)**1.5

