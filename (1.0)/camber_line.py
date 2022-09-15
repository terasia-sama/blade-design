import sys
import numpy as np
import geometric as geom
import curve_modeling as cm
from blade import CambLine


# 调用生成方法
def generate_cl(center_le, center_te, gr_para, airfoil_para):
    npt = gr_para['num_cl']
    mtd = gr_para['cl_mtd']
    if mtd == 'double-arc':
        gen_cl = DbArcCambLine(npt, airfoil_para)
    elif mtd == 'parabola':
        gen_cl = ParabCambLine(npt, airfoil_para)
    elif mtd == 'parametric_design':
        gen_cl = ParaDesiCambLine(center_le, center_te, npt, airfoil_para)
    else:
        print("暂时木有这种中弧线的生成方法，请检查是否有拼写错误或者换个生成方法")
        sys.exit()

    gen_cl.generate()
    cl_tan_vec = gen_cl.return_tangent_vector()
    cl_curva = gen_cl.return_curvature()
    print('cl_generation OVER')
    return gen_cl.cl.curv, cl_tan_vec, cl_curva, mtd


# 中弧线生成方法类
class DbArcCambLine:
    """生成双圆弧中弧线"""
    def __init__(self, npt, airfoil_para):
        self.npt = npt
        self.cl = CambLine(airfoil_para.b)
        self.cl.chi1 = airfoil_para.chi1
        self.cl.chi2 = airfoil_para.chi2
        self.r1 = 0
        self.r2 = 0
        self.cl.curv = np.zeros([self.npt, 2], dtype=float, order='C')
        self.tan_vec = np.zeros([self.npt, 2], dtype=float, order='C')
        self.curva = np.zeros([self.npt], dtype=float, order='C')
        self.flag = 0

    def calc_arc_radius(self):
        """由弦长信息和两圆弧水平相切于最大挠度处求两圆弧半径"""
        self.cl.calc_theta()
        self.r1 = self.cl.b * (1. - np.cos(self.cl.chi2)) / \
                  (np.sin(self.cl.chi1) + np.sin(self.cl.chi2) - np.sin(self.cl.theta))
        self.r2 = self.r1 * (1. - np.cos(self.cl.chi1)) / (1. - np.cos(self.cl.chi2))
        self.cl.a = self.r1 * np.sin(self.cl.chi1)
        self.cl.f_max = self.r1 * (1. - np.cos(self.cl.chi1))

    def generate(self):
        """生成nx2大小的 ndarray 结构的中弧线坐标数据，起始点为前缘点[0, 0]"""
        DbArcCambLine.calc_arc_radius(self)

        center1 = np.array([self.cl.a, -self.r1 * np.cos(self.cl.chi1)])
        npt1 = int(self.npt * self.cl.chi1 / self.cl.theta) + 1
        delta1 = self.cl.chi1 / (npt1 - 1)
        for i in range(npt1):
            self.cl.curv[i, 0] = center1[0] - self.r1 * np.sin(self.cl.chi1 - i * delta1)
            self.cl.curv[i, 1] = center1[1] + self.r1 * np.cos(self.cl.chi1 - i * delta1)
            self.tan_vec[i, :] = np.array([self.cl.curv[i, 1] - center1[1],
                                           -(self.cl.curv[i, 0] - center1[0])])
            self.curva[i] = self.r1
        self.curva[npt1 - 1] = (self.r1 + self.r2) / 2.

        center2 = np.array([self.cl.a, -self.r2 * np.cos(self.cl.chi2)])
        npt2 = self.npt - npt1
        delta2 = self.cl.chi2 / npt2
        for i in range(npt2):
            self.cl.curv[npt1 + i, 0] = center2[0] + self.r2 * np.sin((i + 1) * delta2)
            self.cl.curv[npt1 + i, 1] = center2[1] + self.r2 * np.cos((i + 1) * delta2)
            self.tan_vec[npt1 + i, :] = np.array([self.cl.curv[npt1 + i, 1] - center2[1],
                                                  -(self.cl.curv[npt1 + i, 0] - center2[0])])
            self.curva[npt1 + i] = self.r2
        self.cl.calc_relative_val()
        self.flag = 1

    def return_tangent_vector(self):
        """返回切矢，单位向量，nx2的ndarray数组形式"""
        if self.flag == 0:
            DbArcCambLine.generate(self)
        self.tan_vec = geom.normalization(self.tan_vec)
        return self.tan_vec

    def return_curvature(self):
        """返回曲率，列向量形式"""
        if self.flag == 0:
            DbArcCambLine.generate(self)
        return self.curva


class ParabCambLine:
    """生成抛物线中弧线"""
    def __init__(self, npt, airfoil_para):
        self.npt = npt
        self.cl = CambLine(airfoil_para.b)
        self.cl.chi1 = airfoil_para.chi1
        self.cl.chi2 = airfoil_para.chi2
        self.cl.a = airfoil_para.a
        self.cl.theta = airfoil_para.theta
        self.cl.curv = np.zeros([self.npt, 2], dtype=float, order='C')
        self.tan_vec = np.zeros([self.npt, 2], dtype=float, order='C')
        self.curva = np.zeros([self.npt], dtype=float, order='C')
        self.flag = 0

    def generate(self):
        if self.cl.chi1 != 0 and self.cl.chi2 != 0:
            ParabCambLine.generate_by_edge_angle(self)
        elif self.cl.a != 0 and self.cl.theta != 0:
            ParabCambLine.generate_by_deflection_pos(self)
        self.cl.calc_relative_val()
        self.flag = 1

    def return_tangent_vector(self):
        """返回切矢，单位向量，nx2的ndarray数组形式"""
        if self.flag == 0:
            print('抛物线中弧线尚未生成，请先生成中弧线')
        else:
            N = 1. / np.tan(self.cl.chi2) - 1. / np.tan(self.cl.chi1)
            self.tan_vec[:, 0] = 1.
            self.tan_vec[:, 1] = (self.cl.b - N * self.cl[:, 1] - 2 * self.cl[:, 0]) / \
                                 (N ** 2 * self.cl[:, 1] / 2 + N * self.cl[:, 0] + self.cl.b / np.tan(self.cl.chi1))
            self.tan_vec = geom.normalization(self.tan_vec)
            return self.tan_vec

    def return_curvature(self):
        """返回曲率，列向量形式"""
        if self.flag == 0:
            print('抛物线中弧线尚未生成，请先生成中弧线')
        else:
            N = 1. / np.tan(self.cl.chi2) - 1. / np.tan(self.cl.chi1)
            tan_vec = ParabCambLine.return_tangent_vector(self)
            dy = tan_vec[:, 1] / tan_vec[:, 0]
            d2y = -self.cl.b * (N * dy + 2.) * (1. / np.tan(self.cl.chi1) + N / 2.) / \
                  (N ** 2 * self.cl[:, 1] / 2 + N * self.cl[:, 0] + self.cl.b / np.tan(self.cl.chi1)) ** 2
            self.curva = d2y / (1. + dy ** 2) ** 1.5
            return self.curva

    def generate_by_edge_angle(self):
        """根据前缘角、后缘角和弦长直接生成抛物线"""
        self.cl.calc_theta()

        x = np.arange(0, self.cl.b, self.cl.b / (self.npt - 1))
        N = 1. / np.tan(self.cl.chi2) - 1. / np.tan(self.cl.chi1)
        M = N * x + self.cl.b / np.tan(self.cl.chi1)
        self.cl[:, 0] = x
        self.cl[:, 1] = 2. * (-M + np.sqrt(M ** 2 - N ** 2 * (x - self.cl.b) * x)) / N ** 2
        self.cl.a = self.cl.b / 4 + (self.cl.b - 2) / N / np.tan(self.cl.chi1)
        self.cl.f_max = self.cl.b / 2. / N - 2. * (self.cl.b - 2.) / N ** 2 / np.tan(self.cl.chi1)

    def generate_by_deflection_pos(self):
        """根据最大挠度位置和叶型转角求出前、尾缘角，再生成抛物线"""
        H = (4 * self.cl.a - self.cl.b) / (3 * self.cl.b - 4 * self.cl.a)
        tan1 = (- H - 1 - np.sqrt((H + 1) ** 2 + 4 * H * np.tan(self.cl.theta) ** 2)) / \
               (2 * H * np.tan(self.cl.theta))
        self.cl.chi1 = np.arctan(tan1)
        self.cl.chi2 = self.cl.theta - self.cl.chi1

        x = np.arange(0, self.cl.b, self.cl.b / (self.npt - 1))
        N = 1. / np.tan(self.cl.chi2) - 1. / np.tan(self.cl.chi1)
        M = N * x + self.cl.b / np.tan(self.cl.chi1)
        self.cl[:, 0] = x
        self.cl[:, 1] = 2. * (-M + np.sqrt(M ** 2 - N ** 2 * (x - self.cl.b) * x)) / N ** 2
        self.cl.f_max = (self.cl.b - 2 * self.cl.a) / N


class ParaDesiCambLine:
    """参数化设计中弧线"""
    def __init__(self, center_le, center_te, npt, airfoil_para,): #新增控制点、阶数参数
        self.npt = npt
        self.center_le = center_le
        self.center_te = center_te
        self.cl = CambLine(airfoil_para['chord'])
        self.cl.a = airfoil_para['fle_max_posi']
        self.cl.f_max = airfoil_para['max_fle']
        self.cl.chi1 = airfoil_para['chi1']
        self.cl.chi2 = airfoil_para['chi2']
        self.cl.r1 = airfoil_para['r1']
        self.cl.r2 = airfoil_para['r2']
        self.cl.psi1 = airfoil_para['psi1']
        self.cl.psi2 = airfoil_para['psi2']

        self.cl.curv = np.zeros([self.npt, 2], dtype=float, order='C')
        self.tan_vec = np.zeros([self.npt, 2], dtype=float, order='C')
        self.curva = np.zeros([self.npt], dtype=float, order='C')
        self.flag = 0
        self.cl_ctr_pts = airfoil_para['cl_ctr_pts']
        self.order = airfoil_para['cl_order']




    def generate(self):
        npt = self.npt
        cl_ctr_pts = self.cl_ctr_pts
        order = self.order
        cl = cm.BSplineCurv(npt, cl_ctr_pts, order) #五点三阶B样条
        cl.generate_curve()
        self.tan_vec = cl.return_tangent_vector()
        self.curva = cl.return_curvature()

        self.cl.curv = np.vstack(cl.curv)
        self.tan_vec = np.vstack(self.tan_vec)
        self.curva = np.vstack(self.curva[:, np.newaxis])
        self.flag = 1

    def return_tangent_vector(self):
        """返回切矢，单位向量，nx2的ndarray数组形式"""
        if self.flag == 0:
            print('参数化设计的中弧线尚未生成，请先生成中弧线')
        else:
            return self.tan_vec

    def return_curvature(self):
        """返回曲率，列向量形式"""
        if self.flag == 0:
            print('参数化设计的中弧线尚未生成，请先生成中弧线')
        else:
            return self.curva


# 由ss和ps型线反推中弧线类
class CalcCambLine:
    def __init__(self, num_fitting, ss_pt, ps_pt, ss_mtd, ps_mtd):
        self.ss = ss_pt
        self.ps = ps_pt
        self.nf = num_fitting
        self.ss_mtd = ss_mtd
        self.ps_mtd = ps_mtd
        self.cl = CambLine()
        self.tan_vec = np.array([])
        self.curva = np.array([])
        self.flag = 0

    def return_fitting_profile(self):
        """拟合生成ss和ps加密点，生成各点处向内的法向量，并以元组形式返回"""
        # 拟合加密点生成 及 各点处向内的法向量生成
        if self.ss_mtd == 'cubic_spline':
            csc = cm.CubicSplineCurv(self.nf, self.ss)
            csc.generate_curve()
            ss_curv = csc.curv
            ss_tan_vec = csc.return_tangent_vector()
        elif self.ss_mtd == 'B-spline':
            bsc = cm.BSplineCurv(self.nf, self.ss, 3)
            bsc.generate_curve()
            ss_curv = bsc.curv
            ss_tan_vec = bsc.return_tangent_vector()
        else:
            print('暂时木有这种吸力面曲线的生成方法，请检查是否有拼写错误或者换个生成方法')
            sys.exit()

        if self.ps_mtd == 'cubic_spline':
            csc = cm.CubicSplineCurv(self.nf, self.ps)
            csc.generate_curve()
            ps_curv = csc.curv
            ps_tan_vec = csc.return_tangent_vector()
        elif self.ps_mtd == 'B-spline':
            bsc = cm.BSplineCurv(self.nf, self.ps, 3)
            bsc.generate_curve()
            ps_curv = bsc.curv
            ps_tan_vec = bsc.return_tangent_vector()
        else:
            print('暂时木有这种吸力面曲线的生成方法，请检查是否有拼写错误或者换个生成方法')
            sys.exit()
        # plt.plot(ss_curve[:, 0], ss_curve[:, 1], label='ss_curve')
        # plt.plot(ps_curve[:, 0], ps_curve[:, 1], label='ps_curve')
        # plt.legend()
        # plt.draw()
        # plt.pause(1)
        ss_tan_vec = geom.rotate_points(ss_tan_vec, -np.pi / 2, np.array([0, 0]))
        ps_tan_vec = geom.rotate_points(ps_tan_vec, np.pi / 2, np.array([0, 0]))
        return ss_curv, ps_curv, ss_tan_vec, ps_tan_vec

    def return_max_thickness(self, ss_curv, ps_curv):
        """计算最大厚度及其位置"""
        max_dis = 0.0
        thick_i = 0.00001
        for i in range(ss_curv.shape[0]):
            thick_i = np.linalg.norm(ss_curv[i, :] - ps_curv[0, :])
            for j in range(ps_curv.shape[0]):
                thick_i = min(thick_i, np.linalg.norm(ss_curv[i, :] - ps_curv[j, :]))
            if max_dis < thick_i:
                max_dis = thick_i
                max_dis_pos = [(ss_curv[i, 0] + ps_curv[j, 0])/2, (ss_curv[i, 1] + ps_curv[j, 1])/2]
        print('max_dis = ', max_dis)
        print('max_dis_pos = ', max_dis_pos)
        return max_dis, max_dis_pos

    def equidistant_line_mtd(self):
        """得到中弧线的同时，返回最大圆位置"""
        (ss_curv, ps_curv, ss_tan_vec, ps_tan_vec) = CalcCambLine.return_fitting_profile(self)
        max_dis = CalcCambLine.return_max_thickness(self, ss_curv, ps_curv)[0]
        # 生成等距线簇
        num_step = self.nf
        temp_cl = np.zeros([1, 2], dtype=float, order='C')
        ss_edpt = np.zeros([self.nf, 2], dtype=float, order='C')
        ps_edpt = np.zeros([self.nf, 2], dtype=float, order='C')
        for k in range(num_step):
            # 生成等距线，从最大厚度开始逐渐减小，向两条型线靠拢
            for i in range(self.nf):
                ss_edpt[i, :] = ss_curv[i, :] + (1.0 - k / num_step) * max_dis / 2 * ss_tan_vec[i, :]
                ps_edpt[i, :] = ps_curv[i, :] + (1.0 - k / num_step) * max_dis / 2 * ps_tan_vec[i, :]
                # ss_edpt[i, :] = ss_curv[i, :] + np.cos(k/num_step*np.pi/2) * max_dis/2 * ss_tan_vec[i, :]
                # ps_edpt[i, :] = ps_curv[i, :] + np.cos(k/num_step*np.pi/2) * max_dis/2 * ps_tan_vec[i, :]
            # 判断是否相交
            for i in range(self.nf - 1):
                seg1_a = ss_edpt[i, :]
                seg1_b = ss_edpt[i + 1, :]
                for j in range(self.nf - 1):
                    seg2_c = ps_edpt[j, :]
                    seg2_d = ps_edpt[j + 1, :]
                    intersec = geom.find_seg_intersection(seg1_a, seg1_b, seg2_c, seg2_d)
                    if type(intersec) != int:
                        if 'max_radius' not in dir():
                            max_radius = np.cos(k / num_step * np.pi / 2) * max_dis / 2
                            print('max_radius = ', max_radius)
                            print('intersec = ', intersec)
                            max_radius_pos = intersec
                        intersec = intersec.reshape([1, 2]) #变为二维数组

                        # print('intersec = ', intersec)
                        temp_cl = np.vstack((temp_cl, intersec))
        # 重新排序
        temp_cl = np.delete(temp_cl, 0, axis=0)
        index_sequence = temp_cl[:, 0].argsort()
        self.cl.curv = np.zeros([temp_cl.shape[0], 2], dtype=float, order='C')
        for i in range(temp_cl.shape[0]):
            self.cl.curv[i, :] = temp_cl[index_sequence[i], :]
        print('num of camber line points = ', self.cl.curv.shape[0])

        # 如果输入的ss和ps型线不满足对称关系，补齐
        lambda_s = np.cross(self.ps[0, :] - self.ss[0, :], ps_tan_vec[0, :]) / \
                   np.cross(ss_tan_vec[0, :], ps_tan_vec[0, :])
        lambda_p = np.cross(self.ss[0, :] - self.ps[0, :], ss_tan_vec[0, :]) / \
                   np.cross(ps_tan_vec[0, :], ss_tan_vec[0, :])
        # print('lambda_s = ', lambda_s, 'lambda_p = ', lambda_p)
        if lambda_s - lambda_p < 10 ** (-9) and lambda_s > 0:
            pt0 = self.ss[0, :] + lambda_s * ss_tan_vec[0, :]
            if any(pt0 - self.cl.curv[0, :]) != 0.0:
                self.cl.curv = np.insert(self.cl.curv, 0, pt0, axis=0)
                pt1 = pt0 + np.linalg.norm(self.cl.curv[0, :] - self.cl.curv[1, :]) / 2 * \
                      (ss_tan_vec[0, :] + ps_tan_vec[0, :]) / np.linalg.norm(ss_tan_vec[0, :] + ps_tan_vec[0, :])
                self.cl.curv = np.insert(self.cl.curv, 1, pt1, axis=0)
                print('new num of camber line points = ', self.cl.curv.shape[0])
        self.flag = 1
        # print('cl_calculation OVER')
        return max_radius_pos

    def enumeration_mtd(self):
        (ss_curv, ps_curv, ss_tan_vec, ps_tan_vec) = CalcCambLine.return_fitting_profile(self)
        max_dis = CalcCambLine.return_max_thickness(self, ss_curv, ps_curv)[0]
        # 对吸力面上i点，遍历压力面上所有点j
        n_enum = self.nf * 2
        temp_cl = np.zeros([1, 2], dtype=float, order='C')
        stride = max_dis / 2 / n_enum
        for i in range(ss_curv.shape[0]):
            for k in range(n_enum):
                dis_s = max_dis / 2 * (1 - k * stride)
                center = ss_curv[i, :] + dis_s * ss_tan_vec[i, :]
                dis_p = np.ones(ps_curv.shape[0], dtype=float, order='C') * max_dis
                for j in range(ps_curv.shape[0]):
                    dis_p[j] = np.linalg.norm(center - ps_curv[j, :])
                min_dis_p = min(dis_p)
                if dis_s == min_dis_p:
                    temp_cl = np.vstack((temp_cl, center))
                    break
                elif dis_s < min_dis_p:
                    center = (center + center + stride * ss_tan_vec[i, :]) / 2
                    temp_cl = np.vstack((temp_cl, center))
                    break
        self.cl.curv = np.delete(temp_cl, 0, axis=0)
        print('num of camber line points = ', self.cl.curv.shape[0])
        self.cl.calc_para_by_curv()
        self.flag = 1
        print('cl_calculation OVER')

    def return_tangent_vector(self):
        """返回切矢，单位向量，nx2的ndarray数组形式"""
        if self.flag == 0:
            print('中弧线尚未生成，请先生成中弧线')
        else:
            self.tan_vec = geom.calc_tan_vec_by_difference(self.cl.curv)
            return self.tan_vec

    def return_curva(self):
        if self.flag == 0:
            print('中弧线尚未生成，请先生成中弧线')
        else:
            self.curva = geom.calc_curva_by_difference(self.cl.curv)
            return self.curva
