import numpy as np
import scipy.interpolate as spi
import matplotlib.pyplot as plt
from scipy import integrate
import geometric as geom
import sys
import copy


# 曲线生成库，含3次样条曲线、Beizer曲线、B样条曲线、NURBS曲线等class
class CubicSplineCurv:

    def __init__(self, npt, ctr_pt):
        self.npt = npt
        self.ctr_pt = np.array(ctr_pt)
        self.curv = np.zeros([self.npt, 2], dtype=float, order='C') # 存放曲线，需先调用generate_curve生成曲线
        self.tan_vec = np.zeros([self.npt, 2], dtype=float, order='C')
        self.curva = np.zeros([self.npt], dtype=float, order='C') # 存放曲率，需先调用return_curvature生成曲率
        self.flag = 0

    def generate_curve(self):
        """生成曲线，返回nxm的ndarray数组"""
        # 要求自变量x单调递增,先旋转至水平位置
        angle = geom.angle_of_vec(self.ctr_pt[-1, :] - self.ctr_pt[0, :])
        temp_pt = geom.rotate_points(self.ctr_pt, -angle, self.ctr_pt[0, :])
        # 横坐标分布
        for i in range(self.npt):
            theta = i * np.pi / (self.npt - 1)
            tt = 0.5 * (1.0 - np.cos(theta))  # 两端密，中间疏
            # tt = i / (self.npt - 1) # 均匀分布
            self.curv[i, 0] = tt * (temp_pt[-1, 0] - temp_pt[0, 0]) + temp_pt[0, 0]
        # 插值
        cs = spi.CubicSpline(temp_pt[:, 0], temp_pt[:, 1])
        self.curv[:, 1] = cs(self.curv[:, 0])
        dy = cs(self.curv[:, 0], 1)
        d2y = cs(self.curv[:, 0], 2)
        self.tan_vec[:, 0] = 1.0 / np.sqrt(dy**2 + 1)
        self.tan_vec[:, 1] = dy / np.sqrt(dy**2 + 1)
        self.curva = d2y / (1 + dy**2)**1.5
        # 旋转并返回
        self.curv = geom.rotate_points(self.curv, angle, self.ctr_pt[0, :])
        self.tan_vec = geom.rotate_points(self.tan_vec, angle, [0, 0])
        self.flag = 1

    def return_tangent_vector(self):
        """返回切矢，单位向量，nxm的ndarray数组形式"""
        if self.flag == 0:
            CubicSplineCurv.generate_curve(self)
        return self.tan_vec

    def return_curvature(self):
        """返回曲率，列向量形式"""
        if self.flag == 0:
            CubicSplineCurv.generate_curve(self)
        return self.curva


class BezierCurv:

    def __init__(self, npt, ctr_pt):
        self.npt = npt
        self.ctr_pt = np.array(ctr_pt)
        self.order = self.ctr_pt.shape[0] - 1
        self.dim = self.ctr_pt.shape[1]
        self.curv = np.zeros([self.npt, self.dim], dtype=float, order='C')
        self.tan_vec = np.zeros([self.npt, self.dim], dtype=float, order='C')
        self.curva = np.zeros([self.npt], dtype=float, order='C')
        self.flag = 0

    def generate_curve(self):
        """生成曲线，返回nx2的ndarray数组"""
        delta = np.pi / (self.npt - 1)
        for i in range(self.npt):
            theta = i * delta
            tt = 0.5 * (1.0 - np.cos(theta))  # 分布为两端密，中间疏
            # tt = i / (self.npt - 1)  # 均匀分布

            mid_pt2 = np.zeros([self.order, self.dim], dtype=float, order='C')
            for k in range(self.order):
                n1 = self.order - k
                if k == 0:
                    mid_pt1 = self.ctr_pt
                else:
                    mid_pt1 = mid_pt2

                mid_pt2 = np.zeros([n1, self.dim], dtype=float, order='C')
                for j in range(n1):
                    for m in range(self.dim):
                        mid_pt2[j, m] = (1.0 - tt) * mid_pt1[j, m] + tt * mid_pt1[j + 1, m]
            for m in range(self.dim):
                self.curv[i, m] = mid_pt2[0, m]
        self.flag = 1

    def return_derivative_curve(self, deri_order):
        """求贝塞尔曲线的某阶导矢曲线， nxm的ndarray数组形式"""
        if self.flag == 0:
            BezierCurv.generate_curve(self)
        ctr_pt1 = ctr_pt0 = self.ctr_pt
        coef = 1
        # 递推生成导矢Bezier曲线的控制点
        for i in range(deri_order):
            ctr_pt1 = np.zeros([self.order - i, self.dim], dtype=float, order='C')
            coef *= ctr_pt1.shape[0]
            for j in range(ctr_pt1.shape[0]):
                ctr_pt1[j, :] = ctr_pt0[j+1, :] - ctr_pt0[j, :]
            ctr_pt0 = ctr_pt1
        deri_curv = BezierCurv(self.npt, ctr_pt1)
        deri_curv.generate_curve()
        return coef * deri_curv.curv

    def return_tangent_vector(self):
        """返回切矢，单位向量，nxm的ndarray数组形式"""
        d1_vec = BezierCurv.return_derivative_curve(self, 1)
        for i in range(d1_vec.shape[0]):
            self.tan_vec[i, :] = d1_vec[i, :] / np.linalg.norm(d1_vec[i, :])
        return self.tan_vec

    def return_curvature(self):
        """返回曲率，列向量形式"""
        d1_vec = BezierCurv.return_derivative_curve(self, 1)
        d2_vec = BezierCurv.return_derivative_curve(self, 2)
        if self.dim == 2:
            for i in range(d1_vec.shape[0]):
                self.curva[i] = np.cross(d1_vec[i, :], d2_vec[i, :]) / np.linalg.norm(d1_vec[i, :])**3
        else:
            for i in range(d1_vec.shape[0]):
                temp = np.cross(d1_vec[i, :], d2_vec[i, :]) / np.linalg.norm(d1_vec[i, :])**3
                self.curva[i] = np.linalg.norm(temp)
        return self.curva

    def order_elevation(self, up_order):
        """贝塞尔曲线的升阶性质，对同一条曲线可增加控制点数，相应的阶数也增加，返回增加后的控制点坐标"""
        if type(up_order) != int or up_order < 1:
            print("输入的升阶数有问题，up_order = ", str(up_order))
            sys.exit()
        ctr_pt1 = ctr_pt0 = self.ctr_pt
        # 递推升阶控制点
        for i in range(up_order):
            n = ctr_pt0.shape[0]
            dim = ctr_pt0.shape[1]
            ctr_pt1 = np.zeros([n + 1, dim], dtype=float, order='C')
            ctr_pt1[0] = ctr_pt0[0]
            for j in range(1, n):
                ctr_pt1[j, :] = j/n * ctr_pt0[j-1, :] + (1 - j/n) * ctr_pt0[j, :]
            ctr_pt1[-1, :] = ctr_pt0[-1, :]
            ctr_pt0 = ctr_pt1
        return ctr_pt1


class BSplineCurv:

    def __init__(self, npt, ctr_pt, order, *method):
        self.npt = npt
        self.ctr_pt = np.array(ctr_pt)
        self.order = order
        if method:
            self.mtd = method[0]
        else:
            self.mtd = 'clamped'
        self.dim = self.ctr_pt.shape[1]
        self.nknot = self.ctr_pt.shape[0] + self.order + 1
        self.knot = np.zeros([self.nknot], dtype=float, order='C')
        self.curv = np.zeros([self.npt, self.dim], dtype=float, order='C')
        self.tan_vec = np.zeros([self.npt, self.dim], dtype=float, order='C')
        self.curva = np.zeros([self.npt], dtype=float, order='C')
        self.flag = 0

    def generate_knot(self):
        """根据输入的方法类型字符串，生成[0,1]节点分布，默认为clamped/quasi-uniform分布"""
        nctr_pt = self.ctr_pt.shape[0]
        if self.mtd == 'uniform':
            for k in range(self.nknot):
                self.knot[k] = k / (self.nknot - 1)
        elif self.mtd == 'wrapping control point':
            self.ctr_pt = np.append(self.ctr_pt, self.ctr_pt[:self.order, :], axis=0)
            self.nknot = self.ctr_pt.shape[0] + self.order + 1
            self.knot = np.zeros([self.nknot], dtype=float, order='C')
            for k in range(self.nknot):
                self.knot[k] = k / (self.nknot - 1)
        elif self.mtd == 'wrapping knot':
            self.ctr_pt = np.append(self.ctr_pt, self.ctr_pt[0, :], axis=0)
            self.nknot = self.ctr_pt.shape[0] + self.order + 1
            self.knot = np.zeros([self.nknot], dtype=float, order='C')
            for k in range(nctr_pt):
                self.knot[k] = k / (nctr_pt - 1)
            for k in range(nctr_pt, self.nknot):
                self.knot[k] = self.knot[k - nctr_pt]
        else:
            # 头尾为order+1重节点，与控制点重合，中间为均匀的简单节点
            if (nctr_pt + 1 - self.order) < 2:
                print("比起控制点数，要求的曲线阶数过高，请增加控制点数，或降低阶数，或者改用其他节点分布方式")
                print("当前的节点分布方式为 ", self.mtd)
                sys.exit()
            for k in range(nctr_pt - self.order + 1):
                self.knot[k + self.order] = k * 1.0 / (nctr_pt - self.order)  # 均匀分布
            self.knot[nctr_pt + 1:self.nknot] = 1.0

    def generate_curve(self):
        """生成曲线，返回nx2的ndarray数组"""
        BSplineCurv.generate_knot(self)
        delta = np.pi / (self.npt - 1)
        for i in range(self.npt):
            tt = 0.5 * (1.0 - np.cos(i * delta))  # 中间叶形分布为两端密，中间疏
            # tt = i / (self.npt - 1)  # 均匀分布
            base_fun0 = np.zeros([self.nknot - 1], dtype=float)
            for j in range(self.nknot - 1):
                if self.knot[j] <= tt <= self.knot[j + 1]:
                    base_fun0[j] = 1.0

            for k in range(1, self.order + 1):
                # 计算第k次B-样条基函数
                base_fun1 = np.zeros([self.nknot - 1 - k], dtype=float)
                for j in range(self.nknot - 1 - k):
                    if self.knot[j + k] == self.knot[j]:
                        denominator1 = 1.0
                    else:
                        denominator1 = self.knot[j + k] - self.knot[j]

                    if self.knot[j + k + 1] == self.knot[j + 1]:
                        denominator2 = 1.0
                    else:
                        denominator2 = self.knot[j + k + 1] - self.knot[j + 1]
                    base_fun1[j] = (tt - self.knot[j]) / denominator1 * base_fun0[j] + \
                                   (self.knot[j + k + 1] - tt) / denominator2 * base_fun0[j + 1]
                base_fun0 = base_fun1

            for j in range(self.ctr_pt.shape[0]):
                for m in range(self.dim):
                    self.curv[i, m] += self.ctr_pt[j, m] * base_fun0[j]
        self.flag = 1

    def return_derivative_curve(self, deri_order):
        """求用clamped方法生成的B样条曲线的任意阶导矢曲线， nxm的ndarray数组形式"""
        if self.order - deri_order < 1:
            print("求导阶数 = ", deri_order, "相比该B-样条曲线的阶数 = ", self.order, "而言过大")
            sys.exit()
        if self.mtd == 'quasi-uniform' or self.mtd == 'clamped':
            if self.flag == 0:
                BSplineCurv.generate_knot(self)

            ctr_pt1 = ctr_pt0 = self.ctr_pt
            order0 = self.order
            knot0 = self.knot
            for i in range(deri_order):
                ctr_pt1 = np.zeros([ctr_pt0.shape[0] - 1, self.dim], dtype=float, order='C')
                for j in range(ctr_pt1.shape[0]):
                    for m in range(self.dim):
                        ctr_pt1[j, m] = order0 * (ctr_pt0[j + 1, m] - ctr_pt0[j, m]) / \
                                        (knot0[j + order0 + 1] - knot0[j + 1])
                ctr_pt0 = ctr_pt1
                order0 -= 1
                knot0 = np.delete(knot0, 0, axis=0)
                knot0 = np.delete(knot0, -1, axis=0)
            deri_curv = BSplineCurv(self.npt, ctr_pt1, order0, self.mtd)
            deri_curv.generate_curve()
            return deri_curv.curv
        else:
            print("该方法求导曲线只适用于'quasi-uniform/clamped'方法")
            sys.exit()

    def return_tangent_vector(self):
        """返回切矢，单位向量，nxm的ndarray数组形式"""
        d1_vec = BSplineCurv.return_derivative_curve(self, 1)
        for i in range(d1_vec.shape[0]):
            self.tan_vec[i, :] = d1_vec[i, :] / np.linalg.norm(d1_vec[i, :])
        return self.tan_vec

    def return_curvature(self):
        """返回曲率，列向量形式"""
        d1_vec = BSplineCurv.return_derivative_curve(self, 1)
        d2_vec = BSplineCurv.return_derivative_curve(self, 2)
        if self.dim == 2:
            for i in range(d1_vec.shape[0]):
                self.curva[i] = np.cross(d1_vec[i, :], d2_vec[i, :]) / np.linalg.norm(d1_vec[i, :])**3
        else:
            for i in range(d1_vec.shape[0]):
                temp = np.cross(d1_vec[i, :], d2_vec[i, :]) / np.linalg.norm(d1_vec[i, :])**3
                self.curva[i] = np.linalg.norm(temp)
        return self.curva

    def optimise_curvature_4pts(self, curva):
        if self.ctr_pt.shape[0] != 4:
            print('该方法只适合于4点3阶B样条曲线调整端点曲率用')
            sys.exit()
        v0 = self.ctr_pt[1, :] - self.ctr_pt[0, :]
        v0 = v0 / np.linalg.norm(v0)
        v3 = self.ctr_pt[3, :] - self.ctr_pt[2, :]
        v3 = v3 / np.linalg.norm(v3)
        new_ctr_pt = copy.deepcopy(self.ctr_pt)

        abscurva = map(abs, curva)
        if all(abscurva) != 0:
            kappa = geom.solve_proportion(curva[0], curva[1], v0, -v3,
                                          self.ctr_pt[3, :] - self.ctr_pt[0, :])
            new_ctr_pt[1, :] = self.ctr_pt[0, :] + kappa[0] * v0
            new_ctr_pt[2, :] = self.ctr_pt[3, :] - kappa[1] * v3
        elif curva[0] == 0:
            kappa = np.sqrt(abs(np.cross(-v3, self.ctr_pt[1, :] - self.ctr_pt[3, :]) / 1.5 / curva[1]))
            new_ctr_pt[2, :] = self.ctr_pt[3, :] - kappa * v3
        elif curva[1] == 0:
            kappa = np.sqrt(abs(np.cross(v0, self.ctr_pt[2, :] - self.ctr_pt[0, :]) / 1.5 / curva[0]))
            new_ctr_pt[1, :] = self.ctr_pt[0, :] + kappa * v0
        return new_ctr_pt


class NurbsCurv:

    def __init__(self, npt, ctr_pt, order, weight, *method):
        self.npt = npt
        self.ctr_pt = np.array(ctr_pt)
        self.order = order
        self.w = weight
        if method:
            self.mtd = method[0]
        else:
            self.mtd = 'clamped'
        self.dim = self.ctr_pt.shape[1]
        self.nknot = self.ctr_pt.shape[0] + self.order + 1
        self.knot = np.zeros([self.nknot], dtype=float, order='C')
        self.curv = np.zeros([self.npt, self.dim], dtype=float, order='C')
        self.flag = 0

    def generate_knot(self):
        """根据输入的方法类型字符串，生成[0,1]节点分布，默认为clamped/quasi-uniform分布"""
        nctr_pt = self.ctr_pt.shape[0]
        if self.mtd == 'uniform':
            for k in range(self.nknot):
                self.knot[k] = k / (self.nknot - 1)
        else:
            # 头尾为order+1重节点，与控制点重合，中间为均匀的简单节点
            if (nctr_pt + 1 - self.order) < 2:
                print("比起控制点数，要求的曲线阶数过高，请增加控制点数，或降低阶数，或者改用其他节点分布方式")
                print("当前的节点分布方式为 ", self.mtd)
                sys.exit()
            for k in range(nctr_pt - self.order + 1):
                self.knot[k + self.order] = k * 1.0 / (nctr_pt - self.order)  # 均匀分布
            self.knot[nctr_pt + 1:self.nknot] = 1.0

    def generate_curve(self):
        """生成曲线，返回nx2的ndarray数组"""
        NurbsCurv.generate_knot(self)
        delta = np.pi / (self.npt - 1)
        for i in range(self.npt):
            tt = 0.5 * (1.0 - np.cos(i * delta))  # 中间叶形分布为两端密，中间疏
            # tt = i / (self.npt - 1)  # 均匀分布
            base_fun0 = np.zeros([self.nknot - 1], dtype=float)
            for j in range(self.nknot - 1):
                if self.knot[j] <= tt <= self.knot[j + 1]:
                    base_fun0[j] = 1.0

            for k in range(1, self.order + 1):
                # 计算第k次B-样条基函数
                base_fun1 = np.zeros([self.nknot - 1 - k], dtype=float)
                for j in range(self.nknot - 1 - k):
                    if self.knot[j + k] == self.knot[j]:
                        denominator1 = 1.0
                    else:
                        denominator1 = self.knot[j + k] - self.knot[j]

                    if self.knot[j + k + 1] == self.knot[j + 1]:
                        denominator2 = 1.0
                    else:
                        denominator2 = self.knot[j + k + 1] - self.knot[j + 1]
                    base_fun1[j] = (tt - self.knot[j]) / denominator1 * base_fun0[j] + \
                                   (self.knot[j + k + 1] - tt) / denominator2 * base_fun0[j + 1]
                base_fun0 = base_fun1

            sum_i = 0.0
            for j in range(self.ctr_pt.shape[0]):
                sum_i += self.w(j) * base_fun0[j]
                for m in range(self.dim):
                    self.curv[i, m] += self.w(j) * self.ctr_pt[j, m] * base_fun0[j]
            self.curv[i, :] /= sum_i
        self.flag = 1




def bi_section(a, b, eps, f):
    iter = 0
    print('bi_section method begin:')
    while a < b:
        mid = a + abs(b - a) / 2.0
        if abs(f(mid)) < eps:
            return mid
        if f(mid) * f(b) < 0:
            a = mid
        if f(a) * f(mid) < 0:
            b = mid
        if f(a) * f(mid) >= 0 and f(b) * f(mid) >= 0:
            print('bi_section loop error')
            return 'error'
        iter += 1
        print(str(iter) + '  a= ' + str(a) + ', b= ' + str(b))


def curvature_based_curve3( npt, pt1, k0, k1, curva0, curva1):
    """
     默认起始点为坐标原点，曲线向右逐点生成，曲率为负表示圆心在下
     曲率沿横坐标方向的分布为3点2阶Bezier曲线
    :param pt1:终点坐标
    :param k0, k1: 首尾点斜率
    :param curva0, curva1: 首尾点曲率
    :return:
    """
    print('pt1 = ', pt1, ' k0= ',k0, ' k1= ', k1, ' K0= ', curva0, ' K1= ', curva1)
    def hu(u, xm, x1, K0, K1, Km):
        gu = u**4 * (K0 + K1 - 2.0*Km) * (x1 - 2.0*xm)/2 + \
             2.0/3.0 * u**3 * (2.0*x1*(-K0 + Km) + xm*(5.0*K0 + K1 - 6.0*Km)) + \
             u**2 * (K0*x1 + 2.0*xm*(-2.0*K0 + Km)) + 2.0*u * K0 * xm
        temp = (g0 + gu)**2
        ku = np.sqrt(temp/(1 - temp))
        return ku * (2.0*(x1 - 2.0*xm)*u + 2.0*xm)

    x1 = pt1[0]
    y1 = pt1[1]
    if k0 == np.inf:
        g0 = 1.0
    else:
        g0 = k0 / np.sqrt(1.0 + k0**2)
    g1 = k1 / np.sqrt(1.0 + k1**2)
    print('g0 = ', g0, ' g1 = ', g1)

    xm_lb = 3.0/(curva1 - curva0) * ((curva0 + curva1) / 2 * x1 - (g1 - g0))
    xm_ub = 3.0/(curva1 - curva0) * ((curva0 + 5.0*curva1) / 6 * x1 - (g1 - g0))
    if xm_lb <= 0:
        xm_lb = 0.0
    if xm_ub <= 0:
        print('The given parameters should be adjusted')
    print('xm_lb = ', xm_lb, ' xm_ub = ', xm_ub)

    def calc_err(xm):
        curva_m = 3.0/x1 * ( (curva1 - curva0)/3 * xm + g1 - g0 - (curva0/6 + curva1/2) * x1 )
        result = integrate.quad(hu, 0.0, 1.0, args=(xm, x1, curva0, curva1, curva_m))
        y1_int = result[0]
        print(result[1])
        print('xm=', xm, ' Km= ', curva_m, ' err_y1= ', y1 - y1_int)
        return y1 - y1_int

    print('err_y1(xm_lb) = ', calc_err(xm_lb))
    print('err_y1(xm_ub) = ', calc_err(xm_ub))
    eps = 0.001
    xm = bi_section(xm_lb, xm_ub, eps, calc_err)

    curv = np.zeros([npt, 2], dtype=float, order='C')
    curv[0, :] = np.array([0.0, 0.0])
    for i in range(npt - 1):
        u = (i + 1) / npt
        curva_m = 3.0 / x1 * ((curva1 - curva0) / 3 * xm + g1 - g0 - (curva0 / 6 + curva1 / 2) * x1)
        result = integrate.quad(hu, 0.0, u, args=(xm, x1, curva0, curva1, curva_m))
        curv[i+1, 0] = (x1 - 2.0*xm) * u**2 + 2.0*xm*u
        curv[i+1, 1] = result[0]
    return curv


def curvature_based_curve43( npt, pt1, k0, k1, curva0, curva1, xm, coef_Kn):
    """
     默认起始点为坐标原点，曲线向右逐点生成，曲率为负表示圆心在下
     曲率沿横坐标方向的分布为4点3阶Bezier曲线
    :param pt1:终点坐标
    :param k0, k1: 首尾点斜率
    :param curva0, curva1: 首尾点曲率
    :return:
    """
    print('pt1 = ', pt1, ' k0= ',k0, ' k1= ', k1, ' K0= ', curva0, ' K1= ', curva1)
    def hu(u, xm, xn, x1, K0, Km, Kn, K1):
        gu = u**6 * (-K0 + K1 + 3.0*Km - 3*Kn) * (x1 + 3.0*xm - 3.0*xn)/2 \
             + 3.0/5.0 * u**5 * ((3.0*K0 - 6.0*Km + 3.0*Kn)*x1 + (- 11.0*K0 + 2.0*K1 + 24.0*Km - 15.0*Kn)*xn +
                                 (13.0*K0 - 4.0*K1 - 30.0*Km + 21.0*Kn)*xm) \
             - 3.0/4.0 * u**4 * (3.0*(K0 - Km)*x1 + 3.0*(- 5.0*K0 + 7.0*Km - 2.0*Kn)*xn +
                                 (22.0*K0 - K1 - 36.0*Km + 15.0*Kn)*xm) \
             + u**3 * (K0*x1 + (- 9.0*K0 + 6.0*Km)*xn + (18.0*K0 - 18.0*Km + 3.0*Kn)*xm) \
             + 3.0/2.0 * u**2 * ((-7.0*K0 + 3.0*Km)*xm + 2.0*K0*xn) + 3.0*K0*xm*u
        temp = (g0 + gu)**2
        ku = np.sqrt(temp/(1 - temp))
        return ku * (3.0*(x1 + 3.0*xm - 3.0*xn)*u**2 + 3.0*(-4.0*xm + 2.0*xn)*u + 3.0*xm)

    x1 = pt1[0]
    y1 = pt1[1]
    if k0 == np.inf:
        g0 = 1.0
    else:
        g0 = k0 / np.sqrt(1.0 + k0**2)
    g1 = k1 / np.sqrt(1.0 + k1**2)
    print('g0 = ', g0, ' g1 = ', g1)
    # xm = 0
    # coef_Kn = 42
    curva_n = curva1 * coef_Kn

    def calc_err(xn):
        curva_m = 20.0/3.0/(x1 + xn) * ( g1 - g0 - (curva0/20.0 - curva1/10.0) * 3.0*xn -
                                         (curva0/20.0 + curva1/2.0 + 3.0*curva_n/10.0) * x1 -
                                         (2.0*curva0 - curva1 - curva_n) * 3.0/20.0 * xm)
        result = integrate.quad(hu, 0.0, 1.0, args=(xm, xn, x1, curva0, curva_m, curva_n, curva1))
        y1_int = result[0]
        print('xm=', xm,'  Km=', curva_m, 'xn=', xn,'  Kn=', curva_n, ' err_y1=', y1 - y1_int)
        return y1 - y1_int

    xn_lb = 1.0/3.0/(curva1 - curva0) * ((2.0*curva0 + 5.0*curva1 + 3.0*curva_n) * x1 +
                                         (3.0*curva0 - 1.5*curva1 - 1.5*curva_n) * xm - 10.0*(g1 - g0))
    xn_ub = 1.0/3.0/(curva1 - curva0) * ((curva0 + 13.0*curva1 + 6.0*curva_n) * x1 +
                                         (6.0*curva0 - 3.0*curva1 - 3.0*curva_n) * xm - 20.0*(g1 - g0))
    if xn_lb <= xm:
        xn_lb = 0
    if xn_ub <= 0:
        print('The given parameters should be adjusted')
    xn_lb = xm
    xn_ub = x1
    print('xn_lb=', xn_lb, '  xn_ub=', xn_ub)

    print('err_y1(xn_lb) = ', calc_err(xn_lb))
    print('err_y1(xn_ub) = ', calc_err(xn_ub))
    eps = 1.E-8
    xn = bi_section(xn_lb, xn_ub, eps, calc_err)

    curv = np.zeros([npt, 2], dtype=float, order='C')
    curva = np.zeros([npt], dtype=float, order='C')
    curv[0, :] = np.array([0.0, 0.0])
    curva[0] = curva0


    for i in range(npt - 1):
        u = (i + 1) / npt
        curva_m = 20.0 / 3.0 / (x1 + xn) * (g1 - g0 - (curva0 / 20.0 - curva1 / 10.0) * 3.0 * xn -
                                            (curva0 / 20.0 + curva1 / 2.0 + 3.0 * curva_n / 10.0) * x1 -
                                            (2.0 * curva0 - curva1 - curva_n) * 3.0 / 20.0 * xm)
        result = integrate.quad(hu, 0.0, u, args=(xm, xn, x1, curva0, curva_m, curva_n, curva1))
        curv[i + 1, 0] = 3.0 * u * (1.0 - u)**2 * xm + 3.0 * u**2 * (1.0 - u) * xn + x1 * u**3
        curva[i + 1] = (1 - u)**3 * curva0 + 3.0*u*(1.0 - u)**2 * curva_m + 3.0*u**2*(1.0 - u)*curva_n + u**3*curva1
        curv[i + 1, 1] = result[0]
    print(result[1])
    plt.figure(3)
    plt.plot(curv[:, 0], curva)
    plt.draw()
    return curv


def curvature_based_curve42( npt, pt1, k0, k1, curva0, curva1, xm, coef_Kn, sign, s_length):
    """
     默认起始点为坐标原点，曲线向右逐点生成，曲率为负表示圆心在下
     曲率沿横坐标方向的分布为4点2阶B样条曲线（节点分段函数）
    :param pt1:终点坐标
    :param k0, k1: 首尾点斜率
    :param curva0, curva1: 首尾点曲率
    :return:
    """
    print('pt1 = ', pt1, ' k0= ',k0, ' k1= ', k1, ' K0= ', curva0, ' K1= ', curva1)
    def hu(u, xm, xn, x1, K0, Km, Kn, K1):
        if u <= 0.5:
            gu = - 2.0 * u**4 * (2.0*K0 - 3.0*Km + Kn) * (3.0*xm - xn) + \
                 8.0/3.0 * u**3 * ((8.0*K0 - 9.0*Km + Kn)*xm + 2.0*(Km - K0)*xn) - \
                 2.0 * u**2 * ((7.0*K0 - 4.0*Km)*xm - K0*xn) + 4.0 * u * K0 * xm
        else:
            gu = 2.0 * u**4 * (2.0*K1 + Km - 3.0*Kn) * (2.0*x1 + xm - 3.0*xn) - \
                 8.0/3.0 * u**3 * ((6.0*K1 + 5.0*Km - 11.0*Kn)*x1 + (4.0*K1 + 3.0*Km - 7.0*Kn)*xm +
                                   (- 10.0*K1 - 8.0*Km  + 18.0*Kn)*xn) + \
                 2.0 * u**2 * ((6.0*K1 + 8.0*Km - 12.0*Kn)*x1 + (5.0*K1 + 6.0*Km - 10.0*Kn)*xm +
                               (- 11.0*K1 - 14.0*Km + 22.0*Kn)*xn) + \
                 4.0 * u * (-K1 - 2.0*Km + 2.0*Kn) * (x1 + xm - 2.0*xn) + \
                 (0.5*K1 + 17.0/12.0 * Km - 11.0/12.0 * Kn)*x1 + \
                 (5.0/12.0 * K0 + 7.0/12.0 * K1 + 2.0*Km - Kn)*xm + \
                 (1.0/12.0 *K0 - 13.0/12.0 * K1 - 3.0*Km + 2.0*Kn)*xn

        temp = (g0 + gu)**2
        ku = np.sqrt(temp/(1 - temp))

        if u <= 0.5:
            hu = ku * 4.0 * ((-3.0*xm + xn)*u + xm)
        else:
            hu = ku * 4.0 * ((2.0*x1 + xm - 3.0*xn)*u - x1 - xm + 2.0*xn)
        return hu

    x1 = pt1[0]
    y1 = pt1[1]
    if k0 == np.inf:
        g0 = 1.0
    else:
        g0 = k0 / np.sqrt(1.0 + k0**2)
    g1 = k1 / np.sqrt(1.0 + k1**2)
    print('g0 = ', g0, ' g1 = ', g1)
    # xm = 0
    # coef_Kn = 42
    curva_n = curva1 * coef_Kn

    def calc_err(xn):
        curva_m = 12.0/(x1 + 4.0*xn) * ( g1 - g0 - (5.0*curva0 - curva1 - 4.0*curva_n)/12.0 * xm -
                                         (curva0 - 5.0*curva1)/12.0 * xn - (curva1/2.0 + 5.0*curva_n/12.0) * x1)
        result = integrate.quad(hu, 0.0, 1.0, args=(xm, xn, x1, curva0, curva_m, curva_n, curva1))
        y1_int = result[0]
        print('xm=', xm,'  Km=', curva_m, 'xn=', xn,'  Kn=', curva_n, ' err_y1=', y1 - y1_int)
        return y1 - y1_int

    xn_lb = 1.0/5.0/(curva1 - curva0) * ((curva0 + 6.0*curva1 + 5.0*curva_n) * x1 +
                                         (5.0*curva0 - curva1 - 4.0*curva_n) * xm - 12.0*(g1 - g0))
    xn_ub = 1.0/(5.0*curva1 - curva0 - 4.0*curva_n) * (6.0*(curva1 + curva_n) * x1 +
                                                       (5.0*curva0 - curva1 - 4.0*curva_n) * xm - 12.0*(g1 - g0))
    if xn_lb <= 0 and xm == 0:
        xn_lb = 0.000001
    if xn_lb <= xm:
        temp_a = - (curva0 - 5.0*curva1)/12.0
        temp_b = g1 - g0 - (5.0*curva0 - curva1 - 4.0*curva_n)/12.0 * xm - (curva1/2.0 + 5.0*curva_n/12.0) * x1
        temp_A = 12.0 * temp_a - 4.0 * curva0
        temp_B = 12.0 * temp_b - x1 * curva0 - 4.0 * xm * (curva_n - curva0)
        temp_C = - x1 * xm * (curva_n - curva0)
        Delta = temp_B**2 - 4.0 * temp_A * temp_C
        sol1 = (- temp_B - np.sqrt(Delta))/2.0/temp_A
        sol2 = (- temp_B + np.sqrt(Delta))/2.0/temp_A
        print('sol1 = ', sol1, ' sol2 = ', sol2)
        if sol1 > xm:
            xn_lb = sol1
        else:
            xn_lb = sol2

    if xn_ub <= 0:
        print('The given parameters should be adjusted')
    elif xn_ub >= x1:
        xn_ub = x1
    print('xn_lb=', xn_lb, '  xn_ub=', xn_ub)

    print('err_y1(xn_lb) = ', calc_err(xn_lb))
    print('err_y1(xn_ub) = ', calc_err(xn_ub))
    eps = 1.E-8
    xn = bi_section(xn_lb, xn_ub, eps, calc_err)
    print('xn= ', xn)

    curv = np.zeros([npt, 2], dtype=float, order='C')
    curva = np.zeros([npt], dtype=float, order='C')
    stream = np.zeros([npt + 1], dtype=float, order='C')
    stream[0] = 0.0
    curv[0, :] = np.array([0.0, 0.0])
    curva[0] = curva0
    for i in range(npt - 1):
        # u = (i + 1) / npt
        u = 1.0 - np.cos((i + 1) * np.pi/(2.0*npt))
        curva_m = 12.0 / (x1 + 4.0 * xn) * (g1 - g0 - (5.0 * curva0 - curva1 - 4.0 * curva_n) / 12.0 * xm -
                                            (curva0 - 5.0 * curva1) / 12.0 * xn -
                                            (curva1 / 2.0 + 5.0 * curva_n / 12.0) * x1)
        result = integrate.quad(hu, 0.0, u, args=(xm, xn, x1, curva0, curva_m, curva_n, curva1))
        if u <= 0.5:
            curv[i + 1, 0] = 2.0 * u * (2.0 - 3.0*u) * xm + 2.0 * u**2 * xn
            curva[i + 1] = (1 - 2.0*u)**2 * curva0 + 2.0 * u * (2 - 3.0*u) * curva_m + 2.0*u**2*curva_n
        else:
            curv[i + 1, 0] = 2.0 * (1 - u)**2 * xm + 2.0 * (1 - u)*(3.0*u - 1) * xn + (2.0*u - 1)**2 * x1
            curva[i + 1] = 2.0 * (1 - u)**2 * curva_m + 2.0 * (1 - u)*(3.0*u - 1) * curva_n + (2.0*u - 1)**2 * curva1
        curv[i + 1, 1] = result[0]
        stream[i + 1] = stream[i] + np.linalg.norm(curv[i + 1, :] - curv[i, :])
    stream[-1] = stream[-2] + np.linalg.norm(curv[-1, :] - pt1)
    print(result[1])

    curv_all = np.zeros([npt+1, 2], dtype=float, order='C')
    curva_all = np.zeros([npt+1, 2], dtype=float, order='C')
    curv_all[:-1, :] = curv
    curv_all[-1, :] = pt1
    curva_all[:, 0] = curv_all[:, 0]
    curva_all[:-1, 1] = curva
    curva_all[-1, 1] = curva1
    plt.figure(3)
    legend = sign + '_K0 = ' + str(curva0)
    stream += s_length
    plt.plot(curva_all[:, 0], curva_all[:, 1], label=legend)
    plt.legend()
    plt.draw()

    # 输出记录曲率变化曲线
    sf = lambda x: '{:6.15f}'.format(x)
    with open('D:/Code/python3/BladeProfile/curvature_CDBCurve42_K' + str(curva0) + sign + '.dat', 'w') as out:
        out.write('# leading edge generation method is: CDBCurve_curva0, K0=' + str(curva0) + '\n')
        for i in range(npt + 1):
            out.write(sf(curva_all[i, 0]) + '    ' + sf(curva_all[i, 1]) + '\n')
    return curv


def curvature_based_curve53( npt, pt1, k0, k1, curva0, curva1, xm, coef_Kn, sign, s_length):
    """
     默认起始点为坐标原点，曲线向右逐点生成，曲率为负表示圆心在下
     曲率沿横坐标方向的分布为5点3阶B样条曲线（节点分段函数）
    :param pt1:终点坐标
    :param k0, k1: 首尾点斜率
    :param curva0, curva1: 首尾点曲率
    :return:
    """
    print('pt1 = ', pt1, ' k0= ',k0, ' k1= ', k1, ' K0= ', curva0, ' K1= ', curva1)
    def hu(u, xh, xm, xn, x1, K0, Kh, Km, Kn, K1):
        if u <= 0.5:
            gu = 2.0 * u**6 * (-4.0*K0 + 7.0*Kh - 4.0*Km + Kn) * (7.0*xh - 4.0*xm + xn) + \
                 12.0/5.0 * u**5 * ((66.0*K0 - 105.0*Kh + 45.0*Km - 6.0*Kn)*xh +
                                    (-32.0*K0 + 50.0*Kh - 20.0*Km + 2.0*Kn)*xm + (6.0*K0 - 9.0*Kh + 3.0*Km)*xn) + \
                 3.0 * u**4 * ((-61.0*K0 + 82.0*Kh - 22.0*Km + Kn)*xh +
                               (24.0*K0 - 30.0*Kh + 6.0*Km)*xm + (-3.0*K0 + 3.0*Kh)*xn) + \
                 2.0 * u**3 * ((55.0*K0 - 54.0*Kh + 6.0*Km)*xh + (-16.0*K0 + 12.0*Kh)*xm + K0*xn) + \
                 6.0 * u**2 * (-6.0*K0*xh + 3.0*Kh*xh + K0*xm)  + 6.0 * u * K0*xh
        else:
            gu = 2.0 * u**6 * (-Kh + 4.0*K1 + 4.0*Km - 7.0*Kn) * (-xh + 4.0*x1 + 4.0*xm - 7.0*xn) + \
                 12.0/5.0 * u**5 * ((16.0*Kh - 40.0*K1 - 52.0*Km + 76.0*Kn)*x1 +
                                    (-5.0*Kh + 14.0*K1 + 17.0*Km - 26.0*Kn)*xh +
                                    (18.0*Kh - 48.0*K1 - 60.0*Km + 90.0*Kn)*xm +
                                    (-29.0*Kh + 74.0*K1 + 95.0*Km - 140.0*Kn)*xn) + \
                 3.0 * u**4 * ((-25.0*Kh + 40.0*K1 + 64.0*Km - 79.0*Kn)*x1 +
                               (10.0*Kh - 19.0*K1 - 28.0*Km + 37.0*Kn)*xh +
                               (-32.0*Kh + 56.0*K1 + 86.0*Km - 110.0*Kn)*xm +
                               (47.0*Kh - 77.0*K1 - 122.0*Km  + 152.0*Kn)*xn) + \
                 2.0 * u**3 * ((38.0*Kh - 40.0*K1 - 74.0*Km + 80.0*Kn)*x1 +
                               (-20.0*Kh + 25.0*K1 + 44.0*Km - 50.0*Kn)*xh +
                               (56.0*Kh - 64.0*K1 - 116.0*Km + 128.0*Kn)*xm +
                               (-74.0*Kh + 79.0*K1 + 146.0*Km - 158.0*Kn)*xn) + \
                 6.0 * u**2 * ((5.0*K1 - 7.0*Kh + 10.0*Km - 10.0*Kn)*x1 +
                               (-4.0*K1 + 5.0*Kh - 8.0*Km + 8.0*Kn)*xh +
                               (9.0*K1 - 12.0*Kh + 18.0*Km - 18.0*Kn)*xm +
                               (-10.0*K1 + 14.0*Kh - 20.0*Km + 20.0*Kn)*xn) + \
                 6.0 * u * (2.0*Kh - K1 - 2.0*Km + 2.0*Kn) * (x1 - xh + 2.0*xm - 2.0*xn) + \
                 (-111.0/80.0*Kh + 0.5*K1 + 0.9*Km - 81.0/80.0*Kn)*x1 + \
                 (31.0/80.0*K0 + 2.0*Kh - 49.0/80.0*K1 - 41.0/40.0*Km + 5.0/4.0*Kn)*xh + \
                 (0.1*K0 - 119.0/40.0*Kh + 1.1*K1 + 2.0*Km - 89.0/40.0*Kn)*xm + \
                 (1.0/80.0*K0 + 11.0/4.0*Kh - 79.0/80.0*K1 - 71.0/40.0*Km + 2.0*Kn)*xn

        temp = (g0 + gu)**2
        ku = np.sqrt(temp/(1 - temp))

        if u <= 0.5:
            hu = ku * ((42.0*xh - 24.0*xm + 6.0*xn)*u**2 + ((-36.0*xh + 12.0*xm))*u + 6.0*xh)
        else:
            hu = ku * ((24.0*x1 - 6.0*xh + 24.0*xm - 42.0*xn)*u**2 + (-24.0*x1 + 12.0*xh - 36.0*xm + 48.0*xn)*u +
                       6.0*x1 - 6.0*xh + 12.0*xm - 12.0*xn)
        return hu

    x1 = pt1[0]
    y1 = pt1[1]
    if k0 == np.inf:
        g0 = 1.0
    else:
        g0 = k0 / np.sqrt(1.0 + k0**2)
    g1 = k1 / np.sqrt(1.0 + k1**2)
    print('g0 = ', g0, ' g1 = ', g1)
    # xm = 0
    xh = 0.1*xm
    curva_h = curva0
    curva_n = curva1 * coef_Kn

    def calc_err(xn):
        curva_m = 1.0/(8.0*x1 - 18.0*xh + 18.0*xn)*( 80.0*(g1 - g0) -
                                                     (31.0*curva0 - curva1 - 12.0 * curva_n) * xh -
                                                     (8.0*curva0 - 8.0*curva1 + 18.0*curva_h - 18.0*curva_n) * xm -
                                                     (curva0 + 12.0*curva_h - 31.0*curva1) * xn -
                                                     (curva_h + 40.0*curva1 + 31.0*curva_n) * x1 )
        result = integrate.quad(hu, 0.0, 1.0, args=(xh, xm, xn, x1, curva0, curva_h, curva_m, curva_n, curva1))
        y1_int = result[0]
        print('xm=', xm,'  Km=', curva_m, 'xn=', xn,'  Kn=', curva_n, ' err_y1=', y1 - y1_int)
        return y1 - y1_int

    xn_lb = ( (8.0*curva0 + curva_h + 40.0*curva1 + 31.0*curva_n) * x1 +
              (13.0*curva0 - curva1 - 12.0*curva_n) * curva_h +
              (8.0*curva0 - 8.0*curva1 + 18.0*curva_h - 18.0*curva_n) * xm - 80.0*(g1 - g0)
            )/(31.0*curva1 - 19.0*curva0 - 12.0*curva_h)
    xn_ub = ( (curva_h + 48.0*curva1 + 31.0*curva_n) * x1 +
              (31.0*curva0 - 19.0*curva1 - 12.0*curva_n) * curva_h +
              (8.0*curva0 - 8.0*curva1 + 18.0*curva_h - 18.0*curva_n) * xm - 80.0*(g1 - g0)
            )/(13.0*curva1 - curva0 - 12.0*curva_h)

    if xn_lb <= 0 and xm == 0:
        xn_lb = 0.00001
    if xn_lb <= xm:
        temp_a = - (curva0 - 5.0*curva1)/12.0
        temp_b = g1 - g0 - (5.0*curva0 - curva1 - 4.0*curva_n)/12.0 * xm - (curva1/2.0 + 5.0*curva_n/12.0) * x1
        temp_A = 12.0 * temp_a - 4.0 * curva0
        temp_B = 12.0 * temp_b - x1 * curva0 - 4.0 * xm * (curva_n - curva0)
        temp_C = - x1 * xm * (curva_n - curva0)
        Delta = temp_B**2 - 4.0 * temp_A * temp_C
        sol1 = (- temp_B - np.sqrt(Delta))/2.0/temp_A
        sol2 = (- temp_B + np.sqrt(Delta))/2.0/temp_A
        print('sol1 = ', sol1, ' sol2 = ', sol2)
        if sol1 > xm:
            xn_lb = sol1
        else:
            xn_lb = sol2
    xn_lb = xm

    if xn_ub <= 0:
        print('The given parameters should be adjusted')
    elif xn_ub >= x1:
        xn_ub = x1
    print('xn_lb=', xn_lb, '  xn_ub=', xn_ub)

    print('err_y1(xn_lb) = ', calc_err(xn_lb))
    print('err_y1(xn_ub) = ', calc_err(xn_ub))
    eps = 1.E-8
    xn = bi_section(xn_lb, xn_ub, eps, calc_err)
    print('xn= ', xn)

    curv = np.zeros([npt, 2], dtype=float, order='C')
    curva = np.zeros([npt], dtype=float, order='C')
    stream = np.zeros([npt + 1], dtype=float, order='C')
    stream[0] = 0.0
    curv[0, :] = np.array([0.0, 0.0])
    curva[0] = curva0
    for i in range(npt - 1):
        # u = (i + 1) / npt
        u = 1.0 - np.cos((i + 1) * np.pi/(2.0*npt))
        curva_m = 1.0/(8.0*x1 - 18.0*xm + 18.0*xn) * ( 80.0 * (g1 - g0) -
                                                       (31.0*curva0 - curva1 - 12.0*curva_n)*xh -
                                                       (8.0*curva0 - 8.0*curva1 + 18.0*curva_h - 18.0*curva_n)*xm -
                                                       (curva0 + 12.0*curva_h - 31.0*curva1)*xn -
                                                       (curva_h + 40.0*curva1 + 31.0*curva_n)*x1 )
        result = integrate.quad(hu, 0.0, u, args=(xh, xm, xn, x1, curva0, curva_h, curva_m, curva_n, curva1))
        if u <= 0.5:
            curv[i + 1, 0] = 2.0 * u * (3 - 9.0*u + 7.0*u**2)*xh + 2.0 * u**2 * (3 - 4.0*u)*xm + 2.0 * u **3 * xn
            curva[i + 1] = (1 - 2.0*u)**3 * curva0 + 2.0 * u * (3 - 9.0*u + 7.0*u**2) * curva_h + \
                           2.0 * u**2 * (3 - 4.0*u) * curva_m + 2.0 * u**3 * curva_n
        else:
            curv[i + 1, 0] = 2.0 * (1 - u)**3 * xh + 2.0 * (1 - u)**2 * (4.0 * u - 1)*xm + \
                             2.0 * (1 - u)*(1 - 5.0*u + 7.0 * u**2)*xn + (2.0*u - 1)**3 * x1
            curva[i + 1] = 2.0 * (1 - u)**3 * curva_h + 2.0 * (1 - u)**2 * (4.0*u - 1) * curva_m + \
                           2.0 * (1 - u) * (1 - 5.0*u + 7.0*u**2) * curva_n + (2.0*u - 1)**3 * curva1
        curv[i + 1, 1] = result[0]
        stream[i + 1] = stream[i] + np.linalg.norm(curv[i + 1, :] - curv[i, :])
    stream[-1] = stream[-2] + np.linalg.norm(curv[-1, :] - pt1)
    print(result[1])
    if sign == 'ss' or 'ps':
        curv_all = np.zeros([npt+1, 2], dtype=float, order='C')
        curva_all = np.zeros([npt+1], dtype=float, order='C')
        curv_all[:-1, :] = curv
        curv_all[-1, :] = pt1
        curva_all[:-1] = curva
        curva_all[-1] = curva1
        plt.figure(3)
        legend = 'K0 = ' + str(curva0)
        stream += s_length
        plt.plot(stream, curva_all, label=legend)
        plt.legend()
        plt.draw()
    return curv


def curvature_based_curve52( npt, pt1, k0, k1, curva0, curva1, xm, coef_Kn, sign, s_length):
    """
     默认起始点为坐标原点，曲线向右逐点生成，曲率为负表示圆心在下
     曲率沿横坐标方向的分布为5点2阶B样条曲线（节点分段函数）
    :param pt1:终点坐标
    :param k0, k1: 首尾点斜率
    :param curva0, curva1: 首尾点曲率
    :return:
    """
    print('pt1 = ', pt1, ' k0= ',k0, ' k1= ', k1, ' K0= ', curva0, ' K1= ', curva1)
    def hu(u, xh, xm, xn, x1, K0, Kh, Km, Kn, K1):
        if u <= 1/3:
            gu = - 81.0/8.0 * u**4 * (2.0*K0 - 3.0*Kh + Km) * (3.0*xh - xm) + \
                 9.0 * u**3 * ((8.0*K0 - 9.0*Kh + Km)*xh + 2.0*(Kh - K0)*xm) - \
                 9.0/2.0 * u**2 * ((7.0*K0 - 4.0*Kh)*xh - K0*xm) + 6.0 * u * K0 * xh
        elif 1/3 < u <= 2/3:
            gu = 81.0/8.0 * u**4 * (Kh - 2.0*Km + Kn) * (xh - 2.0*xm + xn) - \
                 9.0/2.0 * u**3 * (Kn*(4.0*xh - 7.0*xm + 3.0*xn) - 2.0*Km*(5.0*xh - 9.0*xm + 4.0*xn) +
                                   Kh*(6.0*xh - 11.0*xm + 5.0*xn)) + \
                 9.0/4.0 * u**2 * (4.0*Kh*(3.0*xh - 5.0*xm + 2.0*xn) - (3.0*Km - Kn)*(5.0*xh - 8.0*xm + 3.0*xn)) - \
                 3.0/2.0 * u * (4.0*Kh - 3.0*Km + Kn) * (2.0*xh - 3.0*xm + xn) + \
                 (5.0/12.0 * K0 + 2.0 * Kh - 17.0/24.0 * Km + 7.0/24.0 * Kn) * xh + \
                 (1.0/12.0 * K0 - 55.0/24.0 * Kh + 9.0/8.0 * Km - 5.0/12.0 * Kn) * xm + \
                 (17.0/24.0 * Kh - Km/3.0 + Kn/8.0) * xn
        else:
            gu = 81.0/8.0 * u**4 * (2.0*K1 + Km - 3.0*Kn) * (2.0*x1 + xm - 3.0*xn) - \
                 9.0/2.0 * u**3 * ((24.0*K1 + 16.0*Km - 40.0*Kn)*x1 + (14.0*K1 + 9.0*Km - 23.0*Kn)*xm +
                                   (-38.0*K1 - 25.0*Km + 63.0*Kn)*xn) + \
                 9.0/4.0 * u**2 * ((48.0*K1 + 42.0*Km - 86.0*Kn)*x1 + (32.0*K1 + 27.0*Km - 57.0*Kn)*xm +
                                   (-80.0*K1 - 69.0*Km + 143.0*Kn)*xn) + \
                 3.0/2.0 * u * (-8.0*K1 - 9.0*Km + 15.0*Kn) * (4.0*x1 + 3.0*xm - 7.0*xn) + \
                 (5.0/12.0 * K0 - 3.0/8.0 * Km - Kn/24.0) * xh + \
                 (K0/12.0 + 20.0/3.0 * K1 + 3.0/8.0 * Kh + 81.0/8.0 * Km - 51.0/4.0 * Kn) * xm + \
                 (-44.0/3.0 * K1 + Kh/24.0 - 21.0 * Km + 225.0/8.0 * Kn) * xn + \
                 (8.0*K1 + 34.0/3.0 * Km - 46.0/3.0 * Kn) * x1

        temp = (g0 + gu)**2
        ku = np.sqrt(temp/(1 - temp))

        if u <= 1/3:
            hu = ku * ((-27.0*xh + 9.0*xm)*u + 6.0*xh)
        elif 1/3 < u <= 2/3:
            hu = ku * ((9.0*xh - 18.0*xm + 9.0*xn)*u - 6.0*xh + 9.0*xm - 3.0*xn)
        else:
            hu = ku * ((18.0*x1 + 9.0*xm - 27.0*xn)*u - 9.0*xm + 21.0*xn - 12.0*x1)
        return hu

    x1 = pt1[0]
    y1 = pt1[1]
    if k0 == np.inf:
        g0 = 1.0
    else:
        g0 = k0 / np.sqrt(1.0 + k0**2)
    g1 = k1 / np.sqrt(1.0 + k1**2)
    print('g0 = ', g0, ' g1 = ', g1)
    # xm = 0
    # coef_Kn = 42
    xh = 0.45*xm
    curva_h = 0.8*curva0
    curva_n = curva1 * coef_Kn

    def calc_err(xn):
        curva_m = 1.0/(2.0*x1 - 9.0*xh + 9.0*xn) * ( 24.0*(g1 - g0) -
                                                     (2.0*curva0 - 2.0*curva1 + 9.0*curva_h - 9.0*curva_n) * xm -
                                                     (10.0*curva0 - curva_n) * xh - (curva_h - 10.0*curva1) * xn -
                                                     (10.0*curva_n + 12.0*curva1) * x1)
        result = integrate.quad(hu, 0.0, 1.0, args=(xh, xm, xn, x1, curva0, curva_h, curva_m, curva_n, curva1))
        y1_int = result[0]
        print('xh=', xh, ' xm=', xm, ' xn=', xn)
        print('Kh=', curva_h,' Km=', curva_m,' Kn=', curva_n, ' err_y1=', y1 - y1_int)
        return y1 - y1_int

    xn_lb = 1.0/(10.0*curva1 - 9.0*curva0 - curva_h) * ((2.0*curva0 + 12.0*curva1 + 10.0*curva_n) * x1 +
                                                        (2.0*curva0 - 2.0*curva1 + 9.0*curva_h - 9.0*curva_n) * xm +
                                                        (curva0 - curva_n) - 24.0*(g1 - g0))
    xn_ub = 1.0/(curva1 - curva_h) * ((14.0*curva1 + 10.0*curva_n) * x1 + (10.0*curva0 - 9.0*curva1 - curva_n) * xh +
                                      (2.0*curva0 - 2.0*curva1 + 3.0*curva_h - 3.0*curva_n) * xm - 24.0*(g1 - g0))
    if xn_lb <= 0 and xm == 0:
        xn_lb = 0.000001
    if xn_lb <= xm:
        temp_a = - (curva0 - 5.0*curva1)/12.0
        temp_b = g1 - g0 - (5.0*curva0 - curva1 - 4.0*curva_n)/12.0 * xm - (curva1/2.0 + 5.0*curva_n/12.0) * x1
        temp_A = 12.0 * temp_a - 4.0 * curva0
        temp_B = 12.0 * temp_b - x1 * curva0 - 4.0 * xm * (curva_n - curva0)
        temp_C = - x1 * xm * (curva_n - curva0)
        Delta = temp_B**2 - 4.0 * temp_A * temp_C
        sol1 = (- temp_B - np.sqrt(Delta))/2.0/temp_A
        sol2 = (- temp_B + np.sqrt(Delta))/2.0/temp_A
        print('sol1 = ', sol1, ' sol2 = ', sol2)
        if sol1 > xm:
            xn_lb = sol1
        else:
            xn_lb = sol2

    if xn_ub <= 0:
        print('The given parameters should be adjusted')
    elif xn_ub >= x1:
        xn_ub = x1
    print('xn_lb=', xn_lb, '  xn_ub=', xn_ub)

    print('err_y1(xn_lb) = ', calc_err(xn_lb))
    print('err_y1(xn_ub) = ', calc_err(xn_ub))
    eps = 1.E-8
    xn = bi_section(xn_lb, xn_ub, eps, calc_err)
    print('xn= ', xn)

    curv = np.zeros([npt, 2], dtype=float, order='C')
    curva = np.zeros([npt], dtype=float, order='C')
    stream = np.zeros([npt + 1], dtype=float, order='C')
    stream[0] = 0.0
    curv[0, :] = np.array([0.0, 0.0])
    curva[0] = curva0
    for i in range(npt - 1):
        # u = (i + 1) / npt
        u = 1.0 - np.cos((i + 1) * np.pi/(2.0*npt))
        curva_m = 1.0/(2.0*x1 - 9.0*xh + 9.0*xn) * ( 24.0*(g1 - g0) -
                                                     (2.0*curva0 - 2.0*curva1 + 9.0*curva_h - 9.0*curva_n) * xm -
                                                     (10.0*curva0 - curva_n) * xh - (curva_h - 10.0*curva1) * xn -
                                                     (10.0*curva_n + 12.0*curva1) * x1)
        result = integrate.quad(hu, 0.0, u, args=(xh, xm, xn, x1, curva0, curva_h, curva_m, curva_n, curva1))
        if u <= 1/3:
            curv[i + 1, 0] = 3.0 * u * (2.0 - 4.5*u) * xh + 4.5 * u**2 * xm
            curva[i + 1] = (1 - 3.0*u)**2 * curva0 + 3.0 * u * (2 - 4.5*u) * curva_h + 4.5*u**2*curva_m
        elif 1/3 < u <= 2/3:
            curv[i + 1, 0] = 1.0/2.0 * (2 - 3.0*u)**2 * xh - 3.0/2.0 * (1 - 6.0*u + 6.0*u**2) * xm + \
                             1.0/2.0 * (3.0*u - 1)**2 * xn
            curva[i + 1] = 1.0/2.0 * (2 - 3.0*u)**2 * curva_h - 3.0/2.0 * (1 - 6.0*u + 6.0*u**2) * curva_m + \
                           1.0/2.0 * (3.0*u - 1)**2 * curva_n
        else:
            curv[i + 1, 0] = 9.0/2.0 * (1 - u)**2 * xm + 3.0/2.0 * (1 - u) * (9.0*u - 5) * xn + (3.0*u - 2)**2 * x1
            curva[i + 1] = 9.0/2.0 * (1 - u)**2 * curva_m + 3.0/2.0 * (1 - u) * (9.0*u - 5) * curva_n +\
                           (3.0*u - 2)**2 * curva1
        curv[i + 1, 1] = result[0]
        stream[i + 1] = stream[i] + np.linalg.norm(curv[i + 1, :] - curv[i, :])
    stream[-1] = stream[-2] + np.linalg.norm(curv[-1, :] - pt1)
    print(result[1])

    curv_all = np.zeros([npt+1, 2], dtype=float, order='C')
    curva_all = np.zeros([npt+1, 2], dtype=float, order='C')
    curv_all[:-1, :] = curv
    curv_all[-1, :] = pt1
    curva_all[:, 0] = curv_all[:, 0]
    curva_all[:-1, 1] = curva
    curva_all[-1, 1] = curva1
    plt.figure(3)
    legend = sign + '_K0 = ' + str(curva0)
    stream += s_length
    plt.plot(curv_all[:, 0], curva_all[:, 1], label=legend)
    plt.legend()
    plt.draw()

    # 输出记录曲率变化曲线
    sf = lambda x: '{:6.15f}'.format(x)
    with open('D:/Code/python3/BladeProfile/curvature_CDBCurve52_K' + str(curva0) + sign + '.dat', 'w') as out:
        out.write('# leading edge generation method is: CDBCurve_curva0, K0=' + str(curva0) + '\n')
        for i in range(npt+1):
            out.write(sf(curva_all[i, 0]) + '    ' + sf(curva_all[i, 1]) + '\n')
    return curv
