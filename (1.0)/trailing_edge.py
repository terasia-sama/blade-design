import sys
import numpy as np
import geometric as geom


# 调用方法
def generate_te(para, airfoil, *end_info, **argdict):
    """
        para: 生成信息
        airfoil: 翼型类，含ss和ps两型面点集的坐标信息
        end_info: ss和ps尾端信息(ss2_vec, ps2_vec, ss2_curva, ps2_curva)
        argdict: 参数字典，key包含'cl2', 'cl2_vec'
    """
    npt = para['num_te']
    if para['te_mtd'] == 'arc':
        te = ArcTE(npt, airfoil)
        r2 = te.generate()
        mtd = para['te_mtd']

    elif para['te_mtd'] == 'ellipse_ratio':
        te = EllipseTE(npt, airfoil, *end_info, **argdict)
        te.generate_by_ratio(para['te_ellipse_ratio'])
        mtd = 'simple_ellipse ratio=' + str(para['te_ellipse_ratio'])
    else:
        print('暂时木有这种尾缘的生成方法，请检查是否有拼写错误或者换个生成方法')
        sys.exit()

    te_curv = {'te_ss': te.te_ss, 'te_ps': te.te_ps, 'te_int': np.array([])}
    # print('te_generation OVER')

    if para['te_mtd'] == 'arc':
        return te_curv, mtd, r2
    else:
        return te_curv, mtd


# 前缘生成方法类
class ArcTE:

    def __init__(self, npt, airfoil, *end_info):
        """
        npt: 生成前缘上分布点数
        airfoil: 翼型类，含ss和ps两型面点集的坐标信息
        end_info: ss和ps首端的切矢组成的元组，若不输入则由坐标差分得到
        """
        self.nte = npt
        self.ss2 = airfoil.ss[-1, :]
        self.ps2 = airfoil.ps[-1, :]
        if end_info:
            self.ss2_vec = end_info[0]
            self.ps2_vec = end_info[1]
        else:
            self.ss2_vec = return_end_vec(airfoil.ss)
            self.ps2_vec = return_end_vec(airfoil.ps)
        self.te_ss = np.array([])
        self.te_ps = np.array([])
        self.radius = 0

    def generate(self):
        (ss_pt, ps_pt, alpha, theta, beta, indicator) = repair_geom(self.ss2, self.ps2,
                                                                    self.ss2_vec, self.ps2_vec)
        self.radius = np.linalg.norm(ps_pt - ss_pt) / 2 / np.cos(beta)
        print('r2=', self.radius)
        temp = np.linalg.norm(ps_pt - ss_pt) / 2 * np.tan(beta)  # 两端点中点与圆心的距离
        self.center = np.zeros([2], dtype=float, order='F')
        self.center[0] = (ss_pt[0] + ps_pt[0]) / 2 - temp * np.cos(theta)
        self.center[1] = (ss_pt[1] + ps_pt[1]) / 2 - temp * np.sin(theta)

        if self.nte % 2 == 0:
            nte_ss = int(self.nte/2 + 1)
        else:
            nte_ss = int((self.nte + 1)/2)
        nte_ps = self.nte + 1 - nte_ss

        te_ss = np.zeros([nte_ss, 2], dtype=float, order='C')
        if indicator == 1:
            nte_ss -= 1
        delta_ss = alpha / nte_ss
        for i in range(nte_ss):
            te_ss[i, 0] = self.center[0] + self.radius * np.cos(i * delta_ss)
            te_ss[i, 1] = self.center[1] + self.radius * np.sin(i * delta_ss)

        te_ps = np.zeros([nte_ps, 2], dtype=float, order='C')
        if indicator == -1:
            nte_ps -= 1
        delta_ps = alpha / nte_ps
        for i in range(nte_ps):
            te_ps[i, 0] = self.center[0] + self.radius * np.cos(i * delta_ps)
            te_ps[i, 1] = self.center[1] - self.radius * np.sin(i * delta_ps)

        # 绕圆心逆时针旋转theta角
        self.te_ss = geom.rotate_points(te_ss, theta, self.center)
        self.te_ps = geom.rotate_points(te_ps, theta, self.center)
        if indicator == 1:
            self.te_ss[-1, :] = ss_pt
        if indicator == -1:
            self.te_ps[-1, :] = ps_pt
        return self.radius
        


class EllipseTE:

    def __init__(self, npt, airfoil, *end_info, **argdict):
        """
        npt: 生成前缘上分布点数
        airfoil: 翼型类，含ss和ps两型面点集的坐标信息
        end_info: ss和ps首端的切矢组成的元组，若不输入则由坐标差分得到
        argdict: 参数字典，key包含'cl2', 'cl2_vec'
        """
        self.nte = npt
        self.ss2 = airfoil.ss[-1, :]
        self.ps2 = airfoil.ps[-1, :]
        if end_info:
            self.ss2_vec = end_info[0]
            self.ps2_vec = end_info[1]
        else:
            self.ss2_vec = return_end_vec(airfoil.ss)
            self.ps2_vec = return_end_vec(airfoil.ps)
        self.te_ss = np.array([])
        self.te_ps = np.array([])
        self.a = 0
        self.b = 0
        self.t = 0
        self.length = 0
        if argdict:
            self.cl2 = argdict['cl2']
            self.cl2_vec = argdict['cl2_vec']
            self.indicator = 1
        else:
            self.indicator = 0

    def generate_by_ratio(self, ratio):
        (ss_pt, ps_pt, alpha, theta, beta, indicator) = repair_geom(self.ss2, self.ps2,
                                                                    self.ss2_vec, self.ps2_vec)
        self.t = np.arctan(1.0 / np.tan(beta) / ratio)  # 参数方程参数t
        self.b = np.linalg.norm(ps_pt - ss_pt) / 2 / np.sin(self.t)
        self.a = self.b * ratio
        temp = self.a * np.cos(self.t)  # 两端点中点与椭圆中心的距离
        center = np.zeros([2], dtype=float, order='F')
        center[0] = (ss_pt[0] + ps_pt[0]) / 2 - temp * np.cos(theta)
        center[1] = (ss_pt[1] + ps_pt[1]) / 2 - temp * np.sin(theta)

        if self.nte % 2 == 0:
            nte_ss = int(self.nte/2 + 1)
        else:
            nte_ss = int((self.nte + 1)/2)
        nte_ps = self.nte + 1 - nte_ss

        te_ss = np.zeros([nte_ss, 2], dtype=float, order='C')
        if indicator == 1:
            nte_ss -= 1
        delta_ss = self.t / nte_ss
        for i in range(nte_ss):
            te_ss[i, 0] = center[0] + self.a * np.cos(i * delta_ss)
            te_ss[i, 1] = center[1] + self.b * np.sin(i * delta_ss)

        te_ps = np.zeros([nte_ps, 2], dtype=float, order='C')
        if indicator == -1:
            nte_ps -= 1
        delta_ps = self.t / nte_ps
        for i in range(nte_ps):
            te_ps[i, 0] = center[0] + self.a * np.cos(i * delta_ps)
            te_ps[i, 1] = center[1] - self.b * np.sin(i * delta_ps)

        # 绕圆心逆时针旋转theta角
        self.te_ss = geom.rotate_points(te_ss, theta, center)
        self.te_ps = geom.rotate_points(te_ps, theta, center)
        if indicator == 1:
            self.te_ss[-1, :] = ss_pt
        if indicator == -1:
            self.te_ps[-1, :] = ps_pt

        temp = np.linalg.norm(ps_pt - ss_pt) / 2 * np.tan(beta)  # 两端点中点与圆心的距离
        cl2 = (ss_pt + ps_pt) / 2 - temp * np.array([np.cos(theta), np.sin(theta)])
        self.length = np.linalg.norm(self.te_ss[0, :] - cl2)


# 公用处理方法
def repair_geom(ss2, ps2, ss2_vec, ps2_vec):
    # 端点切矢与上下两端点连线间的夹角，也等于前缘张角的一半
    alpha_s = geom.angle_between_vec(ss2_vec, ps2 - ss2)
    alpha_p = geom.angle_between_vec(ps2_vec, ss2 - ps2)
    kappa = np.linalg.norm(ps2 - ss2) * np.sin(abs(alpha_p - alpha_s) / 2) / np.sin(
        (alpha_p + alpha_s) / 2)
    if alpha_s < alpha_p:
        ss_pt = ss2 + kappa * ss2_vec
        ps_pt = ps2
        indicator = 1
        print('输入的ss和ps前缘端点不对称，te_ss端点为其补齐')
    elif alpha_s > alpha_p:
        ss_pt = ss2
        ps_pt = ps2 + kappa * ps2_vec
        indicator = -1
        print('输入的ss和ps前缘端点不对称，te_ps端点为其补齐')
    else:
        ss_pt = ss2
        ps_pt = ps2
        indicator = 0
    alpha = (alpha_p + alpha_s) / 2
    theta_s = geom.angle_of_vec(ss2_vec)  # 两切矢的绝对角度
    theta_p = geom.angle_of_vec(ps2_vec)
    theta = (theta_s + theta_p) / 2  # 朝外的前缘进气角进气角方向
    beta = abs(theta_s - theta_p) / 2  # 半夹角
    if beta > np.pi / 2:
        beta -= np.pi / 2
        theta -= np.pi
    return ss_pt, ps_pt, alpha, theta, beta, indicator


def return_end_vec(curv):
    temp = curv[-1, :] - curv[-2, :]
    return temp / np.linalg.norm(temp)


def return_end_curva(curv):
    dy0 = (curv[-1, 1] - curv[-2, 1]) / (curv[-1, 0] - curv[-2, 0])
    dy1 = (curv[-1, 1] - curv[-3, 1]) / (curv[-1, 0] - curv[-3, 0])
    d2y0 = (dy1 - dy0) / (curv[-1, 0] - curv[-3, 0])
    return d2y0 / (1. + dy0**2)**1.5
