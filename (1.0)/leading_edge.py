import sys
import copy
import numpy as np
import geometric as geom
import curve_modeling as cm


# 调用生成方法
def generate_le(gr_para, af_para, airfoil, type, *begin_info, **argdict):
    """
        para: 生成信息
        airfoil: 翼型类，含ss和ps两型面点集的坐标信息
        begin_info: ss和ps首端信息(ss1_vec, ps1_vec, ss1_curva, ps1_curva)
        argdict: 参数字典，key包含'cl1', 'cl1_vec'
    """
    npt = gr_para['num_le']

    # 区分修型还是造型
    if type == 'modify':
        mtd = gr_para['le_mtd']
    elif type == 'model':
        mtd = gr_para['le_model_mtd']

    if mtd == 'arc':
        le = ArcLE(npt, airfoil)
        r1 = le.generate()
    elif mtd == 'ellipse':
        le = EllipseLE(npt, airfoil, *begin_info, **argdict)
        le.generate_by_asymmetric()
        mtd = 'double_ellipse'
    elif mtd == 'ellipse_ratio':
        le = EllipseLE(npt, airfoil, *begin_info, **argdict)
        le.generate_by_ratio(gr_para['le_ellipse_ratio'])
        mtd = 'simple_ellipse ratio=' + str(gr_para['le_ellipse_ratio'])
    elif mtd == 'direct_profile':
        le = DPCurveLE(npt, airfoil, *begin_info, **argdict)
        le.generate(af_para['le_curva'])
    elif mtd == 'CDB':
        le = CDBCurveLE(npt, airfoil, *begin_info, **argdict)
        le.generate(af_para['le_curva'])
    elif mtd == 'Bezier' or 'B-spline' and gr_para['le_order'] == 3:
        le = BSplineLE(npt, airfoil, *begin_info, **argdict)
        #le.generate_3th(af_para['len_LE'], af_para['le_curva'])  #len_LE 键名改过，需核对
        le.generate_by_le_3th(af_para['le_curva'])
        mtd = gr_para['le_model_mtd'] + ' ' + str(gr_para['le_order']) + 'th'

    else:
        print('暂时木有这种前缘的生成方法，请检查是否有拼写错误或者换个生成方法')
        sys.exit()

    le_curv = {'le_ss': le.le_ss, 'le_ps': le.le_ps, 'le_int':np.array([])}
    # print('le_generation OVER')
    if mtd == 'arc':
        return le_curv, mtd, r1
    else:
        return le_curv, mtd


# 前缘生成方法类
class ArcLE:

    def __init__(self, npt, airfoil, *begin_info):
        """
        npt: 生成前缘上分布点数
        airfoil: 翼型类，含ss和ps两型面点集的坐标信息
        end_tan_vec: ss和ps首端的切矢组成的元组，若不输入则由坐标差分得到
        """
        self.nle = npt
        self.ss1 = airfoil.ss[0, :]
        self.ps1 = airfoil.ps[0, :]
        if begin_info:
            self.ss1_vec = -begin_info[0]
            self.ps1_vec = -begin_info[1]
        else:
            self.ss1_vec = -geom.return_begin_vec(airfoil.ss)
            self.ps1_vec = -geom.return_begin_vec(airfoil.ps)
        self.le_ss = np.array([])
        self.le_ps = np.array([])
        self.radius = 0

    def generate(self):
        (ss_pt, ps_pt, alpha, theta, beta, indicator) = repair_geom(self.ss1, self.ps1,
                                                                    self.ss1_vec, self.ps1_vec)
        self.radius = np.linalg.norm(ps_pt - ss_pt) / 2 / np.cos(beta)
        temp = np.linalg.norm(ps_pt - ss_pt) / 2 * np.tan(beta)  # 两端点中点与圆心的距离
        self.center = np.zeros([2], dtype=float, order='F')
        self.center[0] = (ss_pt[0] + ps_pt[0]) / 2 - temp * np.cos(theta)
        self.center[1] = (ss_pt[1] + ps_pt[1]) / 2 - temp * np.sin(theta)

        if self.nle % 2 == 0:
            nle_ss = int(self.nle/2 + 1)
        else:
            nle_ss = int((self.nle + 1)/2)
        nle_ps = self.nle + 1 - nle_ss

        le_ss = np.zeros([nle_ss, 2], dtype=float, order='C')
        if indicator == 1:
            nle_ss -= 1
        delta_ss = alpha / nle_ss
        for i in range(nle_ss):
            le_ss[i, 0] = self.center[0] - self.radius * np.cos(i * delta_ss)
            le_ss[i, 1] = self.center[1] + self.radius * np.sin(i * delta_ss)

        le_ps = np.zeros([nle_ps, 2], dtype=float, order='C')
        if indicator == -1:
            nle_ps -= 1
        delta_ps = alpha / nle_ps
        for i in range(nle_ps):
            le_ps[i, 0] = self.center[0] - self.radius * np.cos(i * delta_ps)
            le_ps[i, 1] = self.center[1] - self.radius * np.sin(i * delta_ps)

        # 绕圆心逆时针旋转theta - pi角
        self.le_ss = geom.rotate_points(le_ss, theta - np.pi, self.center)
        self.le_ps = geom.rotate_points(le_ps, theta - np.pi, self.center)
        if indicator == 1:
            self.le_ss[-1, :] = ss_pt
        if indicator == -1:
            self.le_ps[-1, :] = ps_pt
        return self.radius


class EllipseLE:

    def __init__(self, npt, airfoil, *begin_info, **argdict):
        """
        npt: 生成前缘上分布点数
        airfoil: 翼型类，含ss和ps两型面点集的坐标信息
        end_tan_vec: ss和ps首端的切矢组成的元组，若不输入则由坐标差分得到
        argdict: 参数字典，key包含'cl1', 'cl1_vec'
        """
        self.nle = npt
        self.ss1 = airfoil.ss[0, :]
        self.ps1 = airfoil.ps[0, :]
        if begin_info:
            self.ss1_vec = -begin_info[0]
            self.ps1_vec = -begin_info[1]
        else:
            self.ss1_vec = -geom.return_begin_vec(airfoil.ss)
            self.ps1_vec = -geom.return_begin_vec(airfoil.ps)
        self.le_ss = np.array([])
        self.le_ps = np.array([])
        self.a = 0
        self.b = 0
        self.t = 0
        self.length = 0
        if argdict:
            self.cl1 = argdict['cl1']
            self.cl1_vec = -argdict['cl1_vec']
            self.indicator = 1
        else:
            self.indicator = 0

    def generate_by_ratio(self, ratio):
        (ss_pt, ps_pt, alpha, theta, beta, indicator) = repair_geom(self.ss1, self.ps1,
                                                                    self.ss1_vec, self.ps1_vec)
        self.t = np.arctan(-1.0 / np.tan(beta) / ratio) + np.pi  # 参数方程参数t
        self.b = np.linalg.norm(ps_pt - ss_pt) / 2 / np.sin(self.t)
        self.a = self.b * ratio
        temp = -self.a * np.cos(self.t)  # 两端点中点与椭圆中心的距离
        center = np.zeros([2], dtype=float, order='F')
        center[0] = (ss_pt[0] + ps_pt[0]) / 2 - temp * np.cos(theta)
        center[1] = (ss_pt[1] + ps_pt[1]) / 2 - temp * np.sin(theta)

        if self.nle % 2 == 0:
            nle_ss = int(self.nle/2 + 1)
        else:
            nle_ss = int((self.nle + 1)/2)
        nle_ps = self.nle + 1 - nle_ss

        le_ss = np.zeros([nle_ss, 2], dtype=float, order='C')
        if indicator == 1:
            nle_ss -= 1
        delta_ss = (np.pi - self.t) / nle_ss
        for i in range(nle_ss):
            le_ss[i, 0] = center[0] - self.a * np.cos(i * delta_ss)
            le_ss[i, 1] = center[1] + self.b * np.sin(i * delta_ss)

        le_ps = np.zeros([nle_ps, 2], dtype=float, order='C')
        if indicator == -1:
            nle_ps -= 1
        delta_ps = (np.pi - self.t) / nle_ps
        for i in range(nle_ps):
            le_ps[i, 0] = center[0] - self.a * np.cos(i * delta_ps)
            le_ps[i, 1] = center[1] - self.b * np.sin(i * delta_ps)

        # 绕圆心逆时针旋转theta - pi角
        self.le_ss = geom.rotate_points(le_ss, theta - np.pi, center)
        self.le_ps = geom.rotate_points(le_ps, theta - np.pi, center)
        if indicator == 1:
            self.le_ss[-1, :] = ss_pt
        if indicator == -1:
            self.le_ps[-1, :] = ps_pt

        temp = np.linalg.norm(ps_pt - ss_pt) / 2 * np.tan(beta)  # 两端点中点与圆心的距离
        cl1 = (ss_pt + ps_pt) / 2 - temp * np.array([np.cos(theta), np.sin(theta)])
        self.length = np.linalg.norm(self.le_ss[0, :] - cl1)

    def generate_by_asymmetric(self):
        """
            上下型线斜率不对称时，生成两段非对称椭圆形前缘
        """
        theta_s = geom.angle_of_vec(self.ss1_vec)  # 两切矢的绝对角度
        theta_p = geom.angle_of_vec(self.ps1_vec)

        if self.indicator == 1:
            theta = geom.angle_of_vec(self.cl1_vec)
            beta_s = abs(theta - theta_s)
            beta_p = abs(theta_p - theta)
        else:
            theta = (theta_s + theta_p) / 2  # 朝外的几何进气角方向
            beta_s = beta_p = abs(theta_s - theta_p) / 2  # 半夹角
        if beta_s > np.pi / 2:
            beta_s -= np.pi / 2
            beta_p -= np.pi / 2
            theta -= np.pi

        d = np.dot(self.ps1 - self.ss1, self.cl1_vec)
        h = np.linalg.norm(self.ss1 + d * self.cl1_vec - self.ps1)
        vec_vertical_s = np.array([np.cos(theta + np.pi / 2), np.sin(theta + np.pi / 2)])
        kappa = np.linalg.norm(self.cl1 + np.dot(self.ss1 - self.cl1, self.cl1_vec) * self.cl1_vec - self.ss1) / h
        print('kappa = ', kappa)
        kappa = 0.4945  # 0.49368
        ds = kappa * h
        dp = (1.0 - kappa) * h
        vp_s = self.ss1 + ds * vec_vertical_s
        vp_p = self.ps1 - dp * vec_vertical_s
        # vp_s = self.cl1 + np.dot(self.ss1 - self.cl1, self.cl1_vec) * self.cl1_vec
        # vp_p = self.cl1 + np.dot(self.ps1 - self.cl1, self.cl1_vec) * self.cl1_vec

        m = dp * np.tan(beta_p) / (ds * np.tan(beta_s))

        costs = geom.solve_costs(d, ds, dp, m, beta_s, beta_p)
        t_s = np.arccos(costs)
        a_s = - ds / np.tan(beta_s) * costs / (1 - costs ** 2)
        b_s = ds / np.sin(t_s)
        t_p = np.arccos(m * costs)
        a_p = - dp / np.tan(beta_p) * np.cos(t_p) / (np.sin(t_p)) ** 2
        b_p = dp / np.sin(t_p)
        print('a_s = ', a_s, 'b_s = ', b_s, 'ts = ', t_s)
        print('a_p = ', a_p, 'b_p = ', b_p, 'tp = ', t_p)

        center1 = np.zeros([2], dtype=float, order='F')
        center1[0] = vp_s[0] + a_s * np.cos(t_s) * np.cos(theta)
        center1[1] = vp_s[1] + a_s * np.cos(t_s) * np.sin(theta)
        nle_ss = int(ds / (ds + dp) * self.nle)
        le_ss = np.zeros([nle_ss, 2], dtype=float, order='C')
        delta_ss = (np.pi - t_s) / nle_ss  # 吸力面椭圆弧参数的张角为 pi-ts
        for i in range(nle_ss):
            le_ss[i, 0] = center1[0] - a_s * np.cos(i * delta_ss)
            le_ss[i, 1] = center1[1] + b_s * np.sin(i * delta_ss)

        center2 = np.zeros([2], dtype=float, order='F')
        center2[0] = vp_p[0] + a_p * np.cos(t_p) * np.cos(theta)
        center2[1] = vp_p[1] + a_p * np.cos(t_p) * np.sin(theta)
        nle_ps = self.nle - nle_ss + 1
        le_ps = np.zeros([nle_ps, 2], dtype=float, order='C')
        delta_ps = (np.pi - t_p) / nle_ps  # 吸力面椭圆弧参数的张角为 pi-tp
        for i in range(nle_ps):
            le_ps[i, 0] = center2[0] - a_p * np.cos(i * delta_ps)
            le_ps[i, 1] = center2[1] - b_p * np.sin(i * delta_ps)

        # 绕圆心逆时针旋转theta-pi角
        self.le_ss = geom.rotate_points(le_ss, theta - np.pi, center1)
        self.le_ps = geom.rotate_points(le_ps, theta - np.pi, center2)

        if self.indicator == 1:
            self.length = np.linalg.norm(self.le_ss[0, :] - self.cl1)
        self.a = {'ss': a_s, 'ps': a_p}
        self.b = {'ss': b_s, 'ps': b_p}
        self.t = {'ss': t_s, 'ps': t_p}


class BSplineLE:

    def __init__(self, npt, airfoil, *begin_info, **argdict):
        """
        npt: 生成前缘上分布点数
        airfoil: 翼型类，含ss和ps两型面点集的坐标信息
        end_tan_vec: ss和ps首端的切矢组成的元组，若不输入则由坐标差分得到
        argdict: 参数字典，key包含'cl1', 'cl1_vec'
        """
        self.nle = npt
        self.ss1 = airfoil.ss[0, :]
        self.ps1 = airfoil.ps[0, :]
        if begin_info:
            self.ss1_vec = -begin_info[0]
            self.ps1_vec = -begin_info[1]
            self.ss1_curva = begin_info[2]
            self.ps1_curva = begin_info[3]
        else:
            self.ss1_vec = -geom.return_begin_vec(airfoil.ss)
            self.ps1_vec = -geom.return_begin_vec(airfoil.ps)
            self.ss1_curva = geom.return_begin_curva(airfoil.ss)
            self.ps1_curva = geom.return_begin_curva(airfoil.ps)
        self.le_ss = np.array([])
        self.le_ps = np.array([])
        self.length = 0
        if argdict:
            self.cl1 = argdict['cl1']
            self.cl1_vec = -argdict['cl1_vec']
            self.indicator = 1
        else:
            self.indicator = 0
        #
        self.le_pt = airfoil.le_pt
        self.chi1 = airfoil.para.chi1
        self.psi1 = airfoil.para.psi1

    def generate_3th(self, length, curva0):
        """
            确保曲率连续的3阶贝塞尔曲线，需指定前缘点处曲率
            length: 前缘段长度，上下曲率不对称时，定义为两斜率交点，到其角平分线与两衔接点连线的交点，之间的距离
            curva0: 前缘点处衔接点曲率
        """
        self.length = length
        alpha_s = geom.angle_between_vec(self.ss1_vec, self.ps1 - self.ss1)
        alpha_p = geom.angle_between_vec(self.ps1_vec, self.ss1 - self.ps1)
        theta_s = geom.angle_of_vec(self.ss1_vec)  # 两切矢的绝对角度
        theta_p = geom.angle_of_vec(self.ps1_vec)
        pt0 = np.zeros([2], dtype=float, order='F')  # 前缘点坐标

        if self.indicator == 1:
            theta = geom.angle_of_vec(self.cl1_vec)
            # beta_s = abs(theta - theta_s)
            # beta_p = abs(theta_p - theta)
            pt0 = self.cl1 + length * self.cl1_vec
        else:
            theta = (theta_s + theta_p) / 2  # 朝外的几何进气角方向
            beta = abs(theta_s - theta_p) / 2  # 半夹角
            if beta > np.pi / 2:
                beta -= np.pi / 2
                theta -= np.pi
            if (alpha_s - alpha_p) != 0.0:
                m_pt = self.ss1 + np.sin(beta) / np.cos(abs(alpha_p - alpha_s) / 2) * np.sin(alpha_p) / \
                       np.sin(alpha_p + alpha_s) * (self.ps1 - self.ss1)
            else:
                m_pt = (self.ss1 + self.ps1) / 2
            pt0[0] = m_pt[0] + length * np.cos(theta)
            pt0[1] = m_pt[1] + length * np.sin(theta)

        v0 = np.array([np.cos(theta - np.pi / 2), np.sin(theta - np.pi / 2)])  # 前缘点切线方向，朝吸力面
        # 求解两个控制点在切线方向上的比例系数
        kappa = geom.solve_proportion(curva0, self.ss1_curva, v0, - self.ss1_vec, self.ss1 - pt0)
        ctr_pt_ss = np.zeros([4, 2], dtype=float, order='C')
        ctr_pt_ss[0, :] = pt0
        ctr_pt_ss[1, :] = pt0 + kappa[0] * v0
        ctr_pt_ss[2, :] = self.ss1 - kappa[1] * self.ss1_vec
        ctr_pt_ss[3, :] = self.ss1

        nle_ss = int(self.nle / 2) + 1
        le_ss = cm.BSplineCurv(nle_ss, ctr_pt_ss, 3)
        le_ss.generate_curve()
        self.le_ss = le_ss.curv

        v0 = np.array([np.cos(theta + np.pi / 2), np.sin(theta + np.pi / 2)])  # 前缘点切线方向，朝压力面
        # 求解两个控制点在切线方向上的比例系数
        kappa = geom.solve_proportion(-curva0, -self.ps1_curva, v0, - self.ps1_vec, self.ps1 - pt0)
        ctr_pt_ps = np.zeros([4, 2], dtype=float, order='C')
        ctr_pt_ps[0, :] = pt0
        ctr_pt_ps[1, :] = pt0 + kappa[0] * v0
        ctr_pt_ps[2, :] = self.ps1 - kappa[1] * self.ps1_vec
        ctr_pt_ps[3, :] = self.ps1

        nle_ps = self.nle - nle_ss + 1
        le_ps = cm.BSplineCurv(nle_ps, ctr_pt_ps, 3)
        le_ps.generate_curve()
        self.le_ps = le_ps.curv

    def generate_by_le_3th(self, curva0):
        """
        4控制点3阶B样条曲线clamped分布节点为[0 0 0 0 1 1 1 1],
        对[0,1]上的自变量参数u，迭代后的最终公式为
        Pn = (1-u)**3 * P0 + 3u(1-u)**2 * P1 + 3u**2(1-u) * P2 + u**3 * P3
        """
        self.radius = abs(1.0/curva0)
        v0 = np.array([np.cos(self.chi1 + np.pi/2), np.sin(self.chi1 + np.pi/2)])  # 前缘点切线方向，朝吸力面
        # 求解两个控制点在切线方向上的比例系数
        kappa = geom.solve_proportion(curva0, self.ss1_curva, v0, - self.ss1_vec, self.ss1 - self.le_pt)
        ctr_pt_ss = np.zeros([4, 2], dtype=float, order='C')
        ctr_pt_ss[0, :] = self.le_pt
        ctr_pt_ss[1, :] = self.le_pt + kappa[0] * v0
        ctr_pt_ss[2, :] = self.ss1 - kappa[1] * self.ss1_vec
        ctr_pt_ss[3, :] = self.ss1

        nle_ss = int(self.nle / 2) + 1
        le_ss = cm.BSplineCurv(nle_ss, ctr_pt_ss, 3)
        le_ss.generate_curve()
        self.le_ss = le_ss.curv

        v0 = np.array([np.cos(self.chi1 - np.pi / 2), np.sin(self.chi1 - np.pi / 2)])  # 前缘点切线方向，朝压力面
        # 求解两个控制点在切线方向上的比例系数
        kappa = geom.solve_proportion(-curva0, -self.ps1_curva, v0, - self.ps1_vec, self.ps1 - self.le_pt)
        ctr_pt_ps = np.zeros([4, 2], dtype=float, order='C')
        ctr_pt_ps[0, :] = self.le_pt
        ctr_pt_ps[1, :] = self.le_pt + kappa[0] * v0
        ctr_pt_ps[2, :] = self.ps1 - kappa[1] * self.ps1_vec
        ctr_pt_ps[3, :] = self.ps1

        nle_ps = self.nle - nle_ss + 1
        le_ps = cm.BSplineCurv(nle_ps, ctr_pt_ps, 3)
        le_ps.generate_curve()
        self.le_ps = le_ps.curv

        plt.figure(3)
        curva_ss = le_ss.return_curvature()
        curva_ps = le_ps.return_curvature()
        curva_le = np.hstack([-curva_ss[::-1], curva_ps])
        x1 = le_ss.curv[:, 0]
        x2 = le_ps.curv[:, 0]
        x = np.hstack([-x1[::-1], x2])
        legend = 'curvature=' + str(curva0)
        plt.plot(x, curva_le, label=legend)
        plt.legend()
        plt.ylabel('curvature')

    def generate_by_le_5points_3th(self, curva0):
        """
        5控制点3阶B样条曲线clamped分布节点为[0 0 0 0 0 0.5 1 1 1 1 1],
        对[0,0.5]上的自变量参数u，迭代后的最终公式为
        Pn = (1-2u)**3 * P0 + (6u-18u**2+14u**3) * P1 + (6u**2-8u**3) * P2 + 2u**3 * P3
        对[0.5 1]上的自变量参数u，迭代后的最终公式为
        Pn = 2(1-u)**3 * P1 + 2(1-u)**2(4u-1) * P2 + 2(1-u)(7u**2-5u+1) * P3 + (2u-1)**3 * P4
        """
        self.radius = abs(1.0 / curva0)

    def generate_by_le_4th(self, curva0):
        """
        5控制点4阶B样条曲线clamped分布节点为[0 0 0 0 0 1 1 1 1 1],
        对[0,1]上的自变量参数u，迭代后的最终公式为
        Pn = (1-u)**4 * P0 + 4u(1-u)**3 * P1 + 6u**2(1-u)**2 * P2 + 4u**3(1-u) * P3 + u**4 * P4
        """
        self.radius = abs(1.0 / curva0)


class CDBCurveLE:

    def __init__(self, npt, airfoil, *begin_info, **argdict):
        """
        npt: 生成前缘上分布点数
        airfoil: 翼型类，含ss和ps两型面点集的坐标信息
        end_tan_vec: ss和ps首端的切矢组成的元组，若不输入则由坐标差分得到
        argdict: 参数字典，key包含'cl1', 'cl1_vec'
        """
        self.nle = npt
        self.ss1 = airfoil.ss[0, :]
        self.ps1 = airfoil.ps[0, :]
        if begin_info:
            self.ss1_vec = begin_info[0]
            self.ps1_vec = begin_info[1]
            self.ss1_curva = begin_info[2]
            self.ps1_curva = begin_info[3]
        else:
            self.ss1_vec = geom.return_begin_vec(airfoil.ss)
            self.ps1_vec = geom.return_begin_vec(airfoil.ps)
            self.ss1_curva = geom.return_begin_curva(airfoil.ss)
            self.ps1_curva = geom.return_begin_curva(airfoil.ps)
        self.le_ss = np.array([])
        self.le_ps = np.array([])
        if argdict:
            self.cl1 = argdict['cl1']
            self.cl1_vec = -argdict['cl1_vec']
            self.indicator = 1
        else:
            self.indicator = 0
        #
        self.le_pt = airfoil.le_pt
        self.chi1 = airfoil.para.chi1

    def generate(self, curva0):
        print('le_generation Begin')
        print('【】CDBCurveLE START!')
        pt1 = geom.rotate_points(self.ss1 - self.le_pt, -self.chi1, np.array([0.0, 0.0]))
        temp_vec1 = geom.rotate_points(self.ss1_vec, -self.chi1, np.array([0.0, 0.0]))
        k0 = np.inf
        k1 = temp_vec1[1] / temp_vec1[0]
        nle_ss = int(self.nle/2)
        if curva0 == -3:
            xm = 0.47
        elif curva0 == -3.5:
            xm = 0.21
        elif curva0 == -4:
            xm = 0.1
        else:
            xm = 0
        coef_Kn = 1
        sign ='ss'

        ss_curv = cm.curvature_based_curve42(nle_ss, pt1, k0, k1, curva0, self.ss1_curva, xm, coef_Kn, sign, 0)
        ss_curv = geom.rotate_points(ss_curv, self.chi1, ss_curv[0, :])
        self.le_ss = ss_curv + self.le_pt

        pt1 = geom.rotate_points(self.ps1 - self.le_pt, -self.chi1, np.array([0.0, 0.0]))
        pt1[1] = -pt1[1]
        temp_vec1 = geom.rotate_points(self.ps1_vec, -self.chi1, np.array([0.0, 0.0]))
        k0 = np.inf
        k1 = -temp_vec1[1] / temp_vec1[0]
        nle_ps = self.nle - nle_ss + 1
        if curva0 == -3:
            xm = 0.48
        elif curva0 == -3.5:
            xm = 0.21
        elif curva0 == -4:
            xm = 0.1
        else:
            xm = 0
        coef_Kn = 1
        sign ='ps'
        ps_curv = cm.curvature_based_curve42(nle_ps, pt1, k0, k1, curva0, -self.ps1_curva, xm, coef_Kn, sign, 0)
        ps_curv[:, 1] = -ps_curv[:, 1]
        ps_curv = geom.rotate_points(ps_curv, self.chi1, ps_curv[0, :])
        self.le_ps = ps_curv + self.le_pt

#新加入，直接型线法四点三阶前缘，仍需调试
class DPCurveLE:

    def __init__(self, npt, airfoil, *begin_info, **argdict):
        """
        npt: 生成前缘上分布点数
        airfoil: 翼型类，含ss和ps两型面点集的坐标信息，需要已知ss ps与前缘衔接点的信息
        end_tan_vec: ss和ps首端的切矢组成的元组，若不输入则由坐标差分得到
        argdict: 参数字典，key包含'cl1', 'cl1_vec'
        """
        self.nle = npt
        self.ss1 = airfoil.ss[0, :]
        self.ps1 = airfoil.ps[0, :]
        if begin_info:
            self.ss1_vec = begin_info[0]
            self.ps1_vec = begin_info[1]
            self.ss1_curva = begin_info[2]
            self.ps1_curva = begin_info[3]
        else:
            self.ss1_vec = geom.return_begin_vec(airfoil.ss)
            self.ps1_vec = geom.return_begin_vec(airfoil.ps)
            self.ss1_curva = geom.return_begin_curva(airfoil.ss)
            self.ps1_curva = geom.return_begin_curva(airfoil.ps)
        self.le_ss = np.array([])
        self.le_ps = np.array([])
        if argdict:
            self.cl1 = argdict['cl1']
            self.cl1_vec = -argdict['cl1_vec']
            self.indicator = 1
        else:
            self.indicator = 0
        #
        self.le_pt = airfoil.le_pt
        self.chi1 = airfoil.para.chi1

    def generate(self, curva0):
        print('le_generation Begin')
        print('【】DPCurveLE START!')
        pt1 = geom.rotate_points(self.ss1 - self.le_pt, -self.chi1, np.array([0.0, 0.0]))
        temp_vec1 = geom.rotate_points(self.ss1_vec, -self.chi1, np.array([0.0, 0.0]))
        k0 = np.inf
        k1 = temp_vec1[1] / temp_vec1[0]
        nle_ss = int(self.nle/2)


        curva0 = -4
        print('curva0 = ', curva0)

        if curva0 == -3:
            xm = 0.47
        elif curva0 == -3.5:
            xm = 0.21
        elif curva0 == -4:
            xm = 0.1
        else:
            xm = 0
        xm = 0
        coef_Kn = 1
        ss_curv = cm.curvature_based_curve43(nle_ss, pt1, k0, k1, curva0, self.ss1_curva, xm, coef_Kn)
        ss_curv = geom.rotate_points(ss_curv, self.chi1, ss_curv[0, :])
        self.le_ss = ss_curv + self.le_pt


        pt1 = geom.rotate_points(self.ps1 - self.le_pt, -self.chi1, np.array([0.0, 0.0]))
        pt1[1] = -pt1[1]
        temp_vec1 = geom.rotate_points(self.ps1_vec, -self.chi1, np.array([0.0, 0.0]))
        k0 = np.inf
        k1 = -temp_vec1[1] / temp_vec1[0]
        nle_ps = self.nle - nle_ss + 1

        if curva0 == -3:
            xm = 0.48
        elif curva0 == -3.5:
            xm = 0.21
        elif curva0 == -4:
            xm = 0.1
        else:
            xm = 0

        coef_Kn = 1
        ps_curv = cm.curvature_based_curve43(nle_ps, pt1, k0, k1, curva0, -self.ps1_curva, xm, coef_Kn)
        ps_curv[:, 1] = -ps_curv[:, 1]
        ps_curv = geom.rotate_points(ps_curv, self.chi1, ps_curv[0, :])
        self.le_ps = ps_curv + self.le_pt

class MixMethod:

    def __init__(self, npt, airfoil, *begin_info, **argdict):
        """
        npt: 生成前缘上分布点数
        airfoil: 翼型类，含ss和ps两型面点集的坐标信息
        end_tan_vec: ss和ps首端的切矢组成的元组，若不输入则由坐标差分得到
        argdict: 参数字典，key包含'cl1', 'cl1_vec'
        """
        self.nle = npt
        self.ss1 = airfoil.ss[0, :]
        self.ps1 = airfoil.ps[0, :]
        if begin_info:
            self.ss1_vec = -begin_info[0]
            self.ps1_vec = -begin_info[1]
            self.ss1_curva = begin_info[2]
            self.ps1_curva = begin_info[3]
        else:
            self.ss1_vec = -geom.return_begin_vec(airfoil.ss)
            self.ps1_vec = -geom.return_begin_vec(airfoil.ps)
            self.ss1_curva = geom.return_begin_curva(airfoil.ss)
            self.ps1_curva = geom.return_begin_curva(airfoil.ps)
        self.le_ss = np.array([])
        self.le_ps = np.array([])
        self.a = 0
        self.b = 0
        self.t = 0
        self.length = 0
        self.radius = 0
        if argdict:
            self.cl1 = argdict['cl1']
            self.cl1_vec = -argdict['cl1_vec']
            self.indicator = 1
        else:
            self.indicator = 0
        #
        self.le_pt = airfoil.le_pt
        self.chi1 = airfoil.para.chi1
        self.psi1 = airfoil.para.psi1

# 公用处理方法
def repair_geom(ss1, ps1, ss1_vec, ps1_vec):
    # 端点切矢与上下两端点连线间的夹角，也等于前缘张角的一半
    alpha_s = geom.angle_between_vec(ss1_vec, ps1 - ss1)
    alpha_p = geom.angle_between_vec(ps1_vec, ss1 - ps1)
    kappa = np.linalg.norm(ps1 - ss1) * np.sin(abs(alpha_p - alpha_s) / 2) / np.sin(
        (alpha_p + alpha_s) / 2)
    if alpha_s < alpha_p:
        ss_pt = ss1 + kappa * ss1_vec
        ps_pt = ps1
        indicator = 1
        print('输入的ss和ps前缘端点不对称，le_ss端点为其补齐')
    elif alpha_s > alpha_p:
        ss_pt = ss1
        ps_pt = ps1 + kappa * ps1_vec
        indicator = -1
        print('输入的ss和ps前缘端点不对称，le_ps端点为其补齐')
    else:
        ss_pt = ss1
        ps_pt = ps1
        indicator = 0
    alpha = (alpha_p + alpha_s) / 2
    theta_s = geom.angle_of_vec(ss1_vec)  # 两切矢的绝对角度
    theta_p = geom.angle_of_vec(ps1_vec)
    theta = (theta_s + theta_p) / 2  # 朝外的前缘进气角进气角方向
    beta = abs(theta_s - theta_p) / 2  # 半夹角
    if beta > np.pi / 2:
        beta -= np.pi / 2
        theta -= np.pi
    return ss_pt, ps_pt, alpha, theta, beta, indicator



