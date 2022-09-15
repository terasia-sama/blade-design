# coding:utf-8

import sys
import sympy
import blade
import numpy as np
import pandas as pd
import curve_modeling as cm
import geometric as geom
import matplotlib.pyplot as plt
import scipy.integrate as scint
from camber_line import generate_cl, CalcCambLine
from leading_edge import *
from trailing_edge import generate_te
from direct_profile import generate_ps
from direct_profile import generate_ss


# 定义叶片参数化相关的类，以及实用函数


# 叶片造型类
class AirfoilModeling:

    #  初始输入的原形参数
    def __init__(self, generation_para, airfoil_para):
        '''generation_para存放了各种中弧线、厚度分布等的生成方法'''
        self.gr_para = generation_para
        self.af_para = airfoil_para
        chord_len = (self.af_para['chord'],)
        self.af = blade.AirfoilPara()
        self.mtd = {}
        self.stagger_ang = airfoil_para['stagger_angle']
        self.center_le = airfoil_para['center_le']
        self.center_te = airfoil_para['center_te']
        self.af = blade.Airfoil()
        self.af.b = self.af_para['chord']

    def cambline_thick_mtd(self):
        """中弧线+厚度分布方法"""
        alpha = self.stagger_ang

        center_le = self.center_le
        center_te = self.center_te
        # 生成中弧线
        [self.af.cl, cl_tan_vec, cl_curva, self.mtd['cl_mtd']] = generate_cl(center_le, center_te, self.gr_para,
                                                                             self.af_para)

        self.af.cl = geom.rotate_points(self.af.cl, alpha, np.array([0, 0]))
        cl_tan_vec = geom.rotate_points(cl_tan_vec, alpha, np.array([0, 0]))
        # 生成法线方向
        nor_vec_ss = geom.rotate_points(cl_tan_vec, np.pi / 2, np.array([0, 0]))
        nor_vor_ps = geom.rotate_points(cl_tan_vec, -np.pi / 2, np.array([0, 0]))
        # 生成厚度分布，默认为三次多项式分布
        self.T = ThickDistribution(self.af_para, self.af.cl[:, 0], self.gr_para).Thick.curv
        self.af.ss = self.af.cl + self.T[:, 1][:, np.newaxis] * nor_vec_ss
        self.af.ps = self.af.cl + self.T[:, 1][:, np.newaxis] * nor_vor_ps
        ss_tan_vec = geom.calc_tan_vec_by_difference(self.af.ss)
        ps_tan_vec = geom.calc_tan_vec_by_difference(self.af.ps)
        self.af.ss[-1, :] = [
            self.af.ss[-1, :][0] + self.af_para['r2'] * np.sin(self.af_para['psi2'] * 2) * ss_tan_vec[-1, :][0],
            self.af.ss[-1, :][1] + self.af_para['r2'] * np.sin(self.af_para['psi2'] * 2) * ss_tan_vec[-1, :][1]]
        self.af.ps[-1, :] = [
            self.af.ps[-1, :][0] + self.af_para['r2'] * np.sin(self.af_para['psi2'] * 2) * ps_tan_vec[-1, :][0],
            self.af.ps[-1, :][1] + self.af_para['r2'] * np.sin(self.af_para['psi2'] * 2) * ps_tan_vec[-1, :][1]]
        self.af.ss[0, :] = [
            self.af.ss[0, :][0] - self.af_para['r1'] * np.sin(self.af_para['psi1'] / 2.) * ss_tan_vec[0, :][0],
            self.af.ss[0, :][1] - (self.af_para['r1'] * np.sin(self.af_para['psi1'] / 2.)) * ss_tan_vec[0, :][1]]
        self.af.ps[0, :] = [
            self.af.ps[0, :][0] - self.af_para['r1'] * np.sin(self.af_para['psi1'] / 2.) * ps_tan_vec[0, :][0],
            self.af.ps[0, :][1] - (self.af_para['r1'] * np.sin(self.af_para['psi1'] / 2.)) * ps_tan_vec[0, :][1]]

        self.mtd['ss_mtd'] = self.mtd['ps_mtd'] = 'thickness distribution'
        self.mtd['le_mtd'] = self.mtd['te_mtd'] = 'arc'

        # 生成前后缘
        argdict_le = {'cl1': self.af.cl[0, :], 'cl1_vec': cl_tan_vec[0, :]}
        [self.af.le, self.mtd['le_mtd']] = generate_le(self.gr_para, self.af_para, self.af, argdict_le)
        argdict_te = {'cl1': self.af.cl[-1, :], 'cl1_vec': cl_tan_vec[-1, :]}
        [self.af.te, self.mtd['te_mtd']] = generate_te(self.gr_para, self.af, argdict_te)
        self.af.assemble_profile()
        airfoil = self.af.profile
        return airfoil

    # 新加入，仍需调试
    def direct_profile_mtd(self):
        '''直接型线法'''
        '''需要事先确定的参数有：'''
        alpha = self.stagger_ang
        self.af.le_pt = [0., 0.]
        self.af.ss = np.zeros((self.gr_para['num_ss'], 2), dtype=float, order='C')
        self.af.ps = np.zeros((self.gr_para['num_ss'], 2), dtype=float, order='C')
        self.af.ss[0, :] = self.af_para['ss1']
        self.af.ps[0, :] = self.af_para['ps1']
        self.af.para.chi1 = self.af_para['chi1']

        info_le = (self.af_para['ss1_tan_vec'], self.af_para['ps1_tan_vec'],
                   self.af_para['ss1_curva'], self.af_para['ps1_curva'])
        [self.af.le, self.mtd['le_mtd']] = generate_le(self.gr_para, self.af_para, self.af, 'model', *info_le)

        # 生成吸力面，分三段
        '''
        alpha = self.stagger_ang
        self.af.le_pt = [0., 0.]
        self.af.ss = np.zeros((self.gr_para['num_ss'], 2), dtype=float, order='C')
        self.af.ps = np.zeros((self.gr_para['num_ss'], 2), dtype=float, order='C')
        self.af.ss[0, :] = self.af_para['ss1']
        self.af.ps[0, :] = self.af_para['ps1']
        self.af.para.chi1 = self.af_para['chi1']

        info_le = (self.af_para['ss1_tan_vec'], self.af_para['ps1_tan_vec'],
                   self.af_para['ss1_curva'], self.af_para['ps1_curva'])
        [self.af.le, self.mtd['le_mtd']] = generate_le(self.gr_para, self.af_para, self.af, 'model', *info_le)

        alpha = self.stagger_ang
        self.af.le_pt = [0., 0.]
        self.af.ss = np.zeros((self.gr_para['num_ss'], 2), dtype=float, order='C')
        self.af.ps = np.zeros((self.gr_para['num_ss'], 2), dtype=float, order='C')
        self.af.ss[0, :] = self.af_para['ss1']
        self.af.ps[0, :] = self.af_para['ps1']
        self.af.para.chi1 = self.af_para['chi1']

        info_le = (self.af_para['ss1_tan_vec'], self.af_para['ps1_tan_vec'],
                   self.af_para['ss1_curva'], self.af_para['ps1_curva'])
        [self.af.le, self.mtd['le_mtd']] = generate_le(self.gr_para, self.af_para, self.af, 'model', *info_le)

        alpha = self.stagger_ang
        self.af.le_pt = [0., 0.]
        self.af.ss = np.zeros((self.gr_para['num_ss'], 2), dtype=float, order='C')
        self.af.ps = np.zeros((self.gr_para['num_ss'], 2), dtype=float, order='C')
        self.af.ss[0, :] = self.af_para['ss1']
        self.af.ps[0, :] = self.af_para['ps1']
        self.af.para.chi1 = self.af_para['chi1']

        info_le = (self.af_para['ss1_tan_vec'], self.af_para['ps1_tan_vec'],
                   self.af_para['ss1_curva'], self.af_para['ps1_curva'])
        [self.af.le, self.mtd['le_mtd']] = generate_le(self.gr_para, self.af_para, self.af, 'model', *info_le)
        '''

        # 生成压力面，分三段
        '''
        alpha = self.stagger_ang
        self.af.le_pt = [0., 0.]
        self.af.ss = np.zeros((self.gr_para['num_ss'], 2), dtype=float, order='C')
        self.af.ps = np.zeros((self.gr_para['num_ss'], 2), dtype=float, order='C')
        self.af.ss[0, :] = self.af_para['ss1']
        self.af.ps[0, :] = self.af_para['ps1']
        self.af.para.chi1 = self.af_para['chi1']

        info_le = (self.af_para['ss1_tan_vec'], self.af_para['ps1_tan_vec'],
                   self.af_para['ss1_curva'], self.af_para['ps1_curva'])
        [self.af.le, self.mtd['le_mtd']] = generate_le(self.gr_para, self.af_para, self.af, 'model', *info_le)

        alpha = self.stagger_ang
        self.af.le_pt = [0., 0.]
        self.af.ss = np.zeros((self.gr_para['num_ss'], 2), dtype=float, order='C')
        self.af.ps = np.zeros((self.gr_para['num_ss'], 2), dtype=float, order='C')
        self.af.ss[0, :] = self.af_para['ss1']
        self.af.ps[0, :] = self.af_para['ps1']
        self.af.para.chi1 = self.af_para['chi1']

        info_le = (self.af_para['ss1_tan_vec'], self.af_para['ps1_tan_vec'],
                   self.af_para['ss1_curva'], self.af_para['ps1_curva'])
        [self.af.le, self.mtd['le_mtd']] = generate_le(self.gr_para, self.af_para, self.af, 'model', *info_le)

        alpha = self.stagger_ang
        self.af.le_pt = [0., 0.]
        self.af.ss = np.zeros((self.gr_para['num_ss'], 2), dtype=float, order='C')
        self.af.ps = np.zeros((self.gr_para['num_ss'], 2), dtype=float, order='C')
        self.af.ss[0, :] = self.af_para['ss1']
        self.af.ps[0, :] = self.af_para['ps1']
        self.af.para.chi1 = self.af_para['chi1']

        info_le = (self.af_para['ss1_tan_vec'], self.af_para['ps1_tan_vec'],
                   self.af_para['ss1_curva'], self.af_para['ps1_curva'])
        [self.af.le, self.mtd['le_mtd']] = generate_le(self.gr_para, self.af_para, self.af, 'model', *info_le)
        '''

        self.mtd['ss_mtd'] = self.mtd['ps_mtd'] = 'thickness distribution'
        self.mtd['le_mtd'] = self.mtd['te_mtd'] = 'arc'
        '''

        # 吸力面尾缘
        pt_le_ss = [((len_LE ** 2 + d_LE ** 2) ** 0.5) * np.cos(chi_LE + np.arctan(d_LE / len_LE)), curva_LE_ss]
        LE_ss_ctr_pt = np.array([[],  # 前缘端点及曲率
                                 [],
                                 [],
                                 pt_le_ss])
        LE_ss_curva = cm.BSplineCurv(self.gr_para['num_le'], LE_ss_ctr_pt, 3)
        LE_ss_curva.generate_curve()
        LE_ss_tan_vec = LE_ss_curva.return_tangent_vector()
        LE_ss_curva = LE_ss_curva.curv

        # 压力面尾缘
        pt_le_ps = [((len_LE ** 2 + d_LE ** 2) ** 0.5) * np.cos(chi_LE - np.arctan(d_LE / len_LE)), curva_LE_ps]
        LE_ps_ctr_pt = np.array([[],  # 前缘端点及曲率
                                 [],
                                 [],
                                 pt_le_ps])
        LE_ps_curva = cm.BSplineCurv(self.gr_para['num_le'], LE_ps_ctr_pt, 3)
        LE_ps_curva.generate_curve()
        LE_ps_tan_vec = LE_ps_curva.return_tangent_vector()
        LE_ps_curva = LE_ps_curva.curv
        '''
        self.af.assemble_int_le()
        # self.af.assemble_profile()
        airfoil = self.af.profile
        return airfoil

    def para_design_mtd(self):
        """
        需要事先确定的参数包括： 弦长chord，最大圆弦向位置thick_max_posi[0]，最大圆直径C_max，前缘小圆半径r1，尾缘小圆半径r2，
        叶型前缘角chi1，叶型尾缘角chi2，前缘楔角psi1，尾缘楔角psi2
        """


        # 前缘尾缘方法可改
        # 根据叶型前缘角和前缘楔角生成圆弧形前缘
        AirfoilModeling.arc_le_by_wedge_angle(self)
        # 根据叶型尾缘角和尾缘楔角生成圆弧形尾缘
        AirfoilModeling.arc_te_by_wedge_angle(self)



        # 初始化叶背和叶盆型线
        nss1 = int(self.gr_para['num_ss'] * self.af_para['thick_max_posi'][0] / self.af_para['chord'])
        nss2 = self.gr_para['num_ss'] - nss1
        nps1 = int(self.gr_para['num_ps'] * self.af_para['thick_max_posi'][0] / self.af_para['chord'])
        nps2 = self.gr_para['num_ps'] - nps1
        self.af.ss = np.zeros((self.gr_para['num_ss'], 2), dtype=float, order='C')
        self.af.ps = np.zeros((self.gr_para['num_ps'], 2), dtype=float, order='C')
        self.af.ss[0, :] = np.array([0., 0.]) + self.af_para['r1'] * \
                           np.array([np.cos(np.pi / 2. + self.af_para['chi1'] + self.af_para['psi1'] / 2.),
                                     np.sin(np.pi / 2. + self.af_para['chi1'] + self.af_para['psi1'] / 2.)])
        self.af.ps[0, :] = np.array([0., 0.]) + self.af_para['r1'] * \
                           np.array([np.cos(3. * np.pi / 2. + self.af_para['chi1'] - self.af_para['psi1'] / 2.),
                                     np.sin(3. * np.pi / 2. + self.af_para['chi1'] - self.af_para['psi1'] / 2.)])
        self.af.ss[-1, :] = np.array([self.af_para['chord'], 0.]) + self.af_para['r2'] * \
                            np.array([np.cos(np.pi / 2. - self.af_para['chi2'] - self.af_para['psi2'] / 2.),
                                      np.sin(np.pi / 2. - self.af_para['chi2'] - self.af_para['psi2'] / 2.)])
        self.af.ps[-1, :] = np.array([self.af_para['chord'], 0.]) + self.af_para['r2'] * \
                            np.array([np.cos(-np.pi / 2. - self.af_para['chi2'] + self.af_para['psi2'] / 2.),
                                      np.sin(-np.pi / 2. - self.af_para['chi2'] + self.af_para['psi2'] / 2.)])
        pt_E = np.array([self.af_para['thick_max_posi'][0], self.af_para['C_max']])

        n = self.af_para['r1'] - self.af_para['C_max']
        alpha = -np.pi / 2. + self.af_para['chi1'] + self.af_para['psi1'] / 2.
        m = (pt_E[0] ** 2 + pt_E[1] ** 2 - n ** 2) / 2. / (n / np.cos(alpha) + np.tan(alpha) * pt_E[1] + pt_E[0])
        theta = np.arctan((pt_E[1] - np.tan(alpha) * m) / (pt_E[0] - m))
        pt_S = pt_E + np.array([np.cos(theta), np.sin(theta)])
        k_S = - np.tan(np.pi / 2 - theta)

        n = -self.af_para['r1'] + self.af_para['C_max']
        alpha = 3. * np.pi / 2. + self.af_para['chi1'] - self.af_para['psi1'] / 2.
        m = (pt_E[0] ** 2 + pt_E[1] ** 2 - n ** 2) / 2. / (n / np.cos(alpha) + np.tan(alpha) * pt_E[1] + pt_E[0])
        theta = np.arctan((pt_E[1] - np.tan(alpha) * m) / (pt_E[0] - m))
        pt_P = pt_E - np.array([np.cos(theta), np.sin(theta)])
        k_P = - np.tan(np.pi / 2 - theta)

        pt_S1 = geom.calc_line_intersection(np.tan(self.af_para['chi1'] + self.af_para['psi1'] / 2),
                                            self.af.ss[0, :], k_S, pt_S)
        pt_P1 = geom.calc_line_intersection(np.tan(self.af_para['chi1'] - self.af_para['psi1'] / 2),
                                            self.af.ps[0, :], k_P, pt_P)
        pt_S2 = geom.calc_line_intersection(-np.tan(self.af_para['chi2'] + self.af_para['psi2'] / 2),
                                            self.af.ss[-1, :], k_S, pt_S)
        pt_P2 = geom.calc_line_intersection(-np.tan(self.af_para['chi2'] - self.af_para['psi2'] / 2),
                                            self.af.ps[-1, :], k_P, pt_P)
        ctr_pt_ss1 = cm.BezierCurv(nss1, [self.af.ss[0, :], pt_S1, pt_S]).order_elevation(1)
        ctr_pt_ps1 = cm.BezierCurv(nps1, [self.af.ps[0, :], pt_P1, pt_P]).order_elevation(1)
        ctr_pt_ss2 = cm.BezierCurv(nss2, [pt_S, pt_S2, self.af.ss[-1, :]]).order_elevation(1)
        ctr_pt_ps2 = cm.BezierCurv(nps2, [pt_P, pt_P2, self.af.ps[-1, :]]).order_elevation(1)

        ss1 = cm.BSplineCurv(nss1, ctr_pt_ss1, 3)#4点3阶b样条
        ss1.generate_curve()
        curva_S = ss1.return_curvature()[-1]
        ctr_pt_ss2 = cm.BSplineCurv(nss2 + 1, ctr_pt_ss2, 3).optimise_curvature_4pts([curva_S, 0.])
        ss2 = cm.BSplineCurv(nss2 + 1, ctr_pt_ss2, 3)
        ss2.generate_curve()
        self.af.ss = np.vstack((ss1.curv, np.delete(ss2.curv, 0, axis=0)))

        ps1 = cm.BSplineCurv(nps1, ctr_pt_ps1, 3)
        ps1.generate_curve()
        curva_P = ps1.return_curvature()[-1]
        ctr_pt_ps2 = cm.BSplineCurv(nps2 + 1, ctr_pt_ps2, 3).optimise_curvature_4pts([curva_P, 0.])
        ps2 = cm.BSplineCurv(nps2 + 1, ctr_pt_ps2, 3)
        ps2.generate_curve()
        self.af.ps = np.vstack((ps1.curv, np.delete(ps2.curv, 0, axis=0)))

    def arc_le_by_wedge_angle(self):
        """根据叶型前缘角和前缘楔角生成圆弧形前缘"""
        nle_ss = int(self.gr_para['num_le'] / 2) + 1
        nle_ps = self.gr_para['num_le'] - nle_ss + 1
        self.af.le['le_ss'] = np.zeros([nle_ss, 2], dtype=float, order='C')
        self.af.le['le_ps'] = np.zeros([nle_ps, 2], dtype=float, order='C')
        delta_ss = (np.pi - self.af_para['psi1']) / 2. / nle_ss
        delta_ps = (np.pi - self.af_para['psi1']) / 2. / nle_ps
        for i in range(nle_ss):
            self.af.le['le_ss'][i, :] = np.array([0., 0.]) + self.af_para['r1'] * \
                                        np.array([np.cos(np.pi + self.af_para['chi1'] - i * delta_ss),
                                                  np.sin(np.pi + self.af_para['chi1'] - i * delta_ss)])
        for i in range(nle_ps):
            self.af.le['le_ps'][i, :] = np.array([0., 0.]) + self.af_para['r1'] * \
                                        np.array([np.cos(np.pi + self.af_para['chi1'] + i * delta_ps),
                                                  np.sin(np.pi + self.af_para['chi1'] + i * delta_ps)])

    def arc_te_by_wedge_angle(self):
        """根据叶型尾缘角和尾缘楔角生成圆弧形尾缘"""
        nte_ss = int(self.gr_para['num_te'] / 2) + 1
        nte_ps = self.gr_para['num_te'] - nte_ss + 1
        self.af.te['te_ss'] = np.zeros([nte_ss, 2], dtype=float, order='C')
        self.af.te['te_ps'] = np.zeros([nte_ps, 2], dtype=float, order='C')
        delta_ss = (np.pi - self.af_para['psi2']) / 2. / nte_ss
        delta_ps = (np.pi - self.af_para['psi2']) / 2. / nte_ps
        for i in range(nte_ss):
            self.af.te['te_ss'][i, :] = np.array([self.af_para['chord'], 0.]) + self.af_para['r2'] * \
                                        np.array([np.cos(-self.af_para['chi2'] + i * delta_ss),
                                                  np.sin(-self.af_para['chi2'] + i * delta_ss)])
        for i in range(nte_ps):
            self.af.te['te_ps'][i, :] = np.array([self.af_para['chord'], 0.]) + self.af_para['r2'] * \
                                        np.array([np.cos(-self.af_para['chi2'] - i * delta_ps),
                                                  np.sin(-self.af_para['chi2'] - i * delta_ps)])

    def plot(self):
        # geom.rotate_points(self.origin_af.profile, self.stagger_ang, np.array([0, 0]))
        self.af.assemble_int_le()
        self.af.assemble_int_te()

        fig = plt.figure(2)
        # plt.plot(self.af.cl[:, 0], self.af.cl[:, 1], label = 'cl-curve ' + self.mtd['cl_mtd'] + '_mtd', color = 'b', linewidth = 1, linestyle = '--')
        plt.plot(self.af.ss[:, 0], self.af.ss[:, 1], label='parametric design main profile', color='g')
        plt.plot(self.af.ps[:, 0], self.af.ps[:, 1], color='g')
        plt.plot(self.af.le['le_int'][:, 0], self.af.le['le_int'][:, 1], label='parametric design le+te', color='b')
        plt.plot(self.af.te['te_int'][:, 0], self.af.te['te_int'][:, 1], color='b')

        plt.axis('equal')
        plt.legend(loc='best', frameon=False)
        plt.draw()

        """cl_tan_vec = geom.calc_tan_vec_by_difference(self.af.cl)
        cl_slope = cl_tan_vec[:, 1] / cl_tan_vec[:, 0]
        plt.figure(2)
        plt.plot(self.af.cl[:, 0], cl_slope, label='cl_pd_slope')
        plt.legend()
        plt.draw()"""

        print('airfoil_plot OVER')


# 叶片修型类
# 获取原型数据是通过AirfoilModification类
class AirfoilModification:

    def __init__(self, generation_para, airfoil_para, edge_scope):
        self.gr_para = generation_para
        self.af_para = airfoil_para
        self.edge_scope = edge_scope
        self.raw_data = np.array([])
        self.origin_af = blade.Airfoil()
        self.stagger_ang = 0.
        self.af = blade.Airfoil()
        self.mtd = {}
        self.C_max = 0.
        AirfoilModification.input_data(self, self.gr_para['input_path'])

    def input_data(self, path):
        raw_data = pd.read_table(path, header=None, sep='\s+')
        if all(raw_data.values[0, :] - raw_data.values[-1, :]) == 0.0:
            raw_data = raw_data.iloc[:-1, :]  # 排除首尾点重叠的情况
        cleaned_data = np.zeros([raw_data.shape[0], 2], dtype=float, order='C')
        # 只导入平面叶型二维坐标数据，并将中心平移到原点
        cleaned_data[:, 0] = raw_data.values[:, 0] - raw_data.sum().values[0] / raw_data.shape[0]
        cleaned_data[:, 1] = raw_data.values[:, 1] - raw_data.sum().values[1] / raw_data.shape[0]
        self.raw_data = cleaned_data
        print('data_import OVER')
        AirfoilModification.split_points(self, cleaned_data)

    def split_points(self, points):
        """
            默认上方为吸力面，下方为压力面。左边为前缘，右边为尾缘，否则先做旋转和对称变换
        """
        points = np.array(points)
        num_pt = points.shape[0]
        max_dis = np.zeros([num_pt], dtype='float', order='C')
        point_pair = np.zeros([num_pt], dtype='int', order='C')
        for i in range(num_pt):
            for j in range(num_pt):
                temp_dis = np.linalg.norm(points[i, :] - points[j, :])
                if temp_dis > max_dis[i]:
                    max_dis[i] = temp_dis
                    point_pair[i] = j
        chord0 = max(max_dis)
        print('blade chord of raw data = ', chord0)
        self.chord = chord0
        new_chord = chord0
        '''
        if self.gr_para['new_chord'] and self.gr_para['new_chord'] != 0.0:
            new_chord = self.gr_para['new_chord']
            print('blade chord of new data = ', new_chord)
            points *= new_chord / chord0
        '''

        edge_id = np.argmax(max_dis)
        part1_pt = points[edge_id:point_pair[edge_id] + 1, :]
        part2_pt = points[edge_id::-1, :]
        part2_pt = np.vstack((part2_pt, points[:point_pair[edge_id] - 1:-1, :]))
        if points[edge_id, 0] < points[point_pair[edge_id], 0]:
            le_vertex = points[edge_id, :]
            te_vertex = points[point_pair[edge_id], :]
        else:
            le_vertex = points[point_pair[edge_id], :]
            te_vertex = points[edge_id, :]
            part1_pt = part1_pt[::-1]  # 使顺序为从前缘开始，到尾缘结束
            part2_pt = part2_pt[::-1]

        self.stagger_ang = geom.angle_of_vec(te_vertex - le_vertex)
        part1_pt = geom.rotate_points(part1_pt, -self.stagger_ang, le_vertex) - le_vertex
        part2_pt = geom.rotate_points(part2_pt, -self.stagger_ang, le_vertex) - le_vertex
        print('stagger_angle = ', self.stagger_ang / np.pi * 180)
        print('le_vertex = ', le_vertex)
        print('te_vertex = ', te_vertex)
        if part1_pt.sum(axis=0)[1] / part1_pt.shape[0] > part2_pt.sum(axis=0)[1] / part2_pt.shape[0]:
            ss_half_pt = part1_pt
            ps_half_pt = part2_pt
        else:
            ss_half_pt = part2_pt
            ps_half_pt = part1_pt

        self.origin_af.para.b = new_chord
        self.origin_af.le['le_ss'] = ss_half_pt[0:self.edge_scope['le_ss'], :]
        self.origin_af.le['le_ps'] = ps_half_pt[0:self.edge_scope['le_ps'], :]
        self.origin_af.ss = ss_half_pt[self.edge_scope['le_ss']:(ss_half_pt.shape[0] -
                                                                 self.edge_scope['te_ss']), :]
        self.origin_af.ps = ps_half_pt[self.edge_scope['le_ps']:(ps_half_pt.shape[0] -
                                                                 self.edge_scope['te_ps']), :]
        self.origin_af.te['te_ss'] = ss_half_pt[::-1, :][0:self.edge_scope['te_ss'], :]
        self.origin_af.te['te_ps'] = ps_half_pt[::-1, :][0:self.edge_scope['te_ps'], :]

        [self.origin_af.ss, self.origin_af.ps] = geom.modify_endpoint(self.origin_af.ss, self.origin_af.ps, 'add')
        print('points_split/cleaning OVER')

    @property
    def modify(self):
        gen_cl = CalcCambLine(self.gr_para['num_cl'], self.origin_af.ss, self.origin_af.ps,
                              self.gr_para['ss_mtd'], self.gr_para['ps_mtd'])
        thick_max_posi = gen_cl.equidistant_line_mtd()  # 用等距线簇的方法得到中弧线
        self.mtd['cl_mtd'] = 'equidistant_line'
        self.origin_af.cl = gen_cl.cl.curv
        self.C_max = gen_cl.return_max_thickness(self.origin_af.ss, self.origin_af.ps)[0]
        self.af.cl = gen_cl.cl.curv
        cl_tan_vec = gen_cl.return_tangent_vector()

        # 生成主型线
        [self.af.ss, ss_tan_vec, ss_curva, self.mtd['ss_mtd']] = generate_ss(self.gr_para, self.origin_af.ss)
        # np.savetxt('D:/SJTU/Master22/Blade design/result/fit_ss_tan_vec.txt', ss_tan_vec, fmt='%0.8f')
        # np.savetxt('D:/SJTU/Master22/Blade design/result/fit_ss_curva.txt', ss_curva, fmt='%0.8f')
        [self.af.ps, ps_tan_vec, ps_curva, self.mtd['ps_mtd']] = generate_ps(self.gr_para, self.origin_af.ps)
        # np.savetxt('D:/SJTU/Master22/Blade design/result/fit_ps_tan_vec.txt', ss_tan_vec, fmt='%0.8f')
        # np.savetxt('D:/SJTU/Master22/Blade design/result/fit_ps_curva.txt', ps_curva, fmt='%0.8f')
        self.psi1 = geom.angle_between_vec(ss_tan_vec[0, :], ps_tan_vec[0, :])
        self.psi2 = geom.angle_between_vec(ss_tan_vec[-1, :], ps_tan_vec[-1, :])

        # 新加入
        self.chi1 = geom.angle_between_vec(cl_tan_vec[0, :], [1, 0])
        self.chi2 = geom.angle_between_vec(cl_tan_vec[-1, :], [1, 0])
        # 以下参数仍需修改
        self.d_LE = np.linalg.norm(self.af.ss[0, :]) * np.sin(
            np.arctan(self.af.ss[0, 1] / self.af.ss[0, 0] - self.chi1))
        self.len_LE = np.linalg.norm(self.af.ss[0, :]) * np.cos(
            np.arctan(self.af.ss[0, 1] / self.af.ss[0, 0] - self.chi1))
        self.d_TE = np.linalg.norm(self.af.ss[-1, :]) * np.sin(
            np.arctan(self.af.ss[-1, 1] / self.af.ss[-1, 0] - self.chi2))
        self.len_TE = np.linalg.norm(self.af.ss[-1, :]) * np.cos(
            np.arctan(self.af.ss[-1, 1] / self.af.ss[-1, 0] - self.chi2))

        print('psi1 of raw data = ', self.psi1)
        print('psi2 of raw data = ', self.psi2)
        print('chi1 of raw data = ', self.chi1)
        print('chi2 of raw data = ', self.chi2)
        print('d_LE of raw data = ', self.d_LE)
        print('len_LE of raw data = ', self.len_LE)
        print('d_LE of raw data = ', self.d_TE)
        print('len_LE of raw data = ', self.len_TE)

        [f_max_posi, max_fle] = max(self.af.cl, key=lambda x: x[1])



        info_le = (ss_tan_vec[0, :], ps_tan_vec[0, :], ss_curva[0], ps_curva[0])
        argdict_le = {'cl1': self.af.cl[0, :], 'cl1_vec': cl_tan_vec[0, :]}
        if self.gr_para['le_mtd'] == 'arc':
            [self.af.le, self.mtd['le_mtd'], r1] = generate_le(self.gr_para, self.af_para, self.af, 'modify', *info_le,
                                                       **argdict_le)
        else:
            [self.af.le, self.mtd['le_mtd']] = generate_le(self.gr_para, self.af_para, self.af, 'modify', *info_le,
                                                       **argdict_le)

        info_te = (ss_tan_vec[-1, :], ps_tan_vec[-1, :], ss_curva[-1], ps_curva[-1])
        argdict_te = {'cl1': self.af.cl[-1, :], 'cl1_vec': cl_tan_vec[-1, :]}
        if self.gr_para['te_mtd'] == 'arc':
            [self.af.te, self.mtd['te_mtd'], r2] = generate_te(self.gr_para, self.af, 'modify', *info_te, **argdict_te)
        else:
            [self.af.te, self.mtd['te_mtd']] = generate_te(self.gr_para, self.af, 'modify', *info_te, **argdict_te)
        # self.stagger_ang = self.gr_para['stagger_angle']

        # nor_vec_ss = geom.rotate_points(cl_tan_vec, np.pi/2, np.array([0, 0]))
        # self.T = np.zeros([327, 2], dtype=float, order='C')
        # self.T = (self.af.ss - self.af.cl)/nor_vec_ss
        # np.savetxt('C:/Users/Iris/Desktop/BladeProfile/thick.txt', self.T, fmt='%0.8f')

        print('airfoil_modification OVER')

        if self.stagger_ang != 0:
            temp_af = self.af
            self.af = blade.Cascade()
            self.af.inherit_airfoil_data(temp_af)
            self.af.para.stagger_ang = self.stagger_ang
            self.af.para.calc_geom_inout_angle()
            self.af.install('CCW', np.array([0, 0]))
            print('cascade_modification OVER')

            temp_af = self.origin_af
            self.origin_af = blade.Cascade()
            self.origin_af.inherit_airfoil_data(temp_af)
            self.origin_af.para.stagger_ang = self.stagger_ang
            self.origin_af.para.calc_geom_inout_angle()
            self.origin_af.install('CCW', np.array([0, 0]))

        """self.T = np.zeros_like(self.af.cl, dtype = float, order='C')
        nor_vec_ss = geom.rotate_points(cl_tan_vec, np.pi/2, np.array([0, 0]))
        self.T = (self.af.ss - self.af.cl) / nor_vec_ss
        self.T[:, 0] = self.af.cl[:, 0]
        self.T[:, 1] = np.abs(self.T[:, 1])"""

        print('【le_curva_ss】', geom.return_begin_curva(self.af.le['le_ss']), '【le_curva_ps】', geom.return_begin_curva(self.af.le['le_ps']))
        # 输出很臃肿，考虑用字典传,update()方法
        af_para_update = {'max_fle': max_fle, 'fle_max_posi': [f_max_posi, max_fle], 'psi1':self.psi1, 'psi2':self.psi2,
                          'chi1':self.chi1, 'chi2':self.chi2, 'd_LE':self.d_LE, 'd_TE': self.d_TE, 'len_LE':self.len_LE,
                          'len_TE':self.len_TE, 'chord':self.chord, 'C_max':self.C_max, 'thick_max_posi':thick_max_posi,
                          'ss1_tan_vec':ss_tan_vec[0, :], 'ss1_curva': ss_curva[0],
                          'ps1_tan_vec':ps_tan_vec[0, :], 'ps1_curva':ps_curva[0],
                          'ss1': self.af.ss[0, :], 'ps1':self.af.ps[0, :],
                          'ss2_tan_vec': ss_tan_vec[-1, :], 'ss2_curva': ss_curva[-1],
                          'ps2_tan_vec': ps_tan_vec[-1, :], 'ps2_curva': ps_curva[-1],
                          'ss2': self.af.ss[-1, :], 'ps2': self.af.ps[-1, :],
                          'le_curva': -geom.return_begin_curva(self.af.le['le_ss'])}
        # 前缘尾缘半径
        if self.gr_para['le_mtd'] == 'arc':
            af_para_update['r1'] = r1
        if self.gr_para['te_mtd'] == 'arc':
            af_para_update['r2'] = r2
        print('r1=',r1,'r2=',r2)

        return af_para_update, gen_cl.cl.curv, cl_tan_vec

    def plot(self):
        self.origin_af.assemble_profile()
        raw_airfoil = self.origin_af.profile
        # geom.rotate_points(self.origin_af.profile, self.stagger_ang, np.array([0, 0]))
        self.af.assemble_int_le()
        self.af.assemble_int_te()

        plt.figure(2, figsize=(5, 5))
        plt.scatter(raw_airfoil[:, 0], raw_airfoil[:, 1], label='original airfoil', color='orange')
        plt.axis('equal')
        plt.legend(loc='best', frameon=False)
        plt.draw()

        """cl_tan_vec = geom.calc_tan_vec_by_difference(self.af.cl)
        cl_slope = cl_tan_vec[:, 1] / cl_tan_vec[:, 0]
        plt.figure(2)
        plt.plot(self.af.cl[:, 0], cl_slope, label='cl_slope')
        plt.legend()
        plt.draw()"""
        print('airfoil_plot OVER')


# 叶型输出
class OutputData:

    def __init__(self, output_info, data, **mtd):
        self.format = output_info['format']
        self.data = data
        self.mtd = mtd
        self.path = output_info['output_path'] + '/blade_' + mtd['cl_mtd'] + '.' + self.format

        if isinstance(data, blade.Airfoil):
            self.kind = 'A'
        elif isinstance(data, blade.Cascade):
            self.kind = 'C'
        elif isinstance(data, blade.Blade):
            self.kind = 'B'

        if self.kind == ('A' or 'C'):
            self.span = output_info['span']
            self.mid_r = output_info['mid_radius']
            self.hub = self.mid_r - self.span / 2
            self.shroud = self.mid_r + self.span / 2

    def output(self):
        if self.format == 'curve':
            if self.kind == ('A' or 'C'):
                OutputData.curve_2t3(self)
            if self.kind == 'B':
                OutputData.curve_3(self)
        print('data_output OVER')

    def curve_2t3(self):
        self.data.assemble_profile()
        sf0 = lambda x: '{:3.6f}'.format(x)
        sf = lambda x: '{:6.15f}'.format(x)

        with open(self.path, 'w') as out:
            out.write('# leading edge generation method is: ' + self.mtd['le_mtd'] + '\n')
            if self.kind == 'C':
                out.write('# stagger angle = ' + sf0(self.data.stagger_ang) +
                          ',    inlet angle = ' + sf0(self.data.beta_1k) +
                          ',    outlet angle = ' + sf0(self.data.beta_2k) + '\n')
            out.write('# Profile 1 at 0.00000 \n')
            for i in range(self.data.profile.shape[0]):
                out.write(sf(self.data.profile[i, 0]) + '    ' +
                          sf(self.data.profile[i, 1]) + '    ' + sf(self.hub) + '\n')
            out.write('\n# Profile 2 at 50.00000 \n')
            for i in range(self.data.profile.shape[0]):
                out.write(sf(self.data.profile[i, 0]) + '    ' +
                          sf(self.data.profile[i, 1]) + '    ' + sf(self.mid_r) + '\n')
            out.write('\n# Profile 3 at 100.00000 \n')
            for i in range(self.data.profile.shape[0]):
                out.write(sf(self.data.profile[i, 0]) + '    ' +
                          sf(self.data.profile[i, 1]) + '    ' + sf(self.shroud) + '\n')
            out.close()
        print('output OVER')

    def curve_3(self):
        for i in range(len(self.data.profile)):
            self.data.profile[i][1].assemble_profile()
        sf0 = lambda x: '{:3.6f}'.format(x)
        sf = lambda x: '{:6.15f}'.format(x)

        with open(self.path, 'w') as out:
            out.write('# leading edge generation method is: ' + self.mtd['le_mtd'] + '\n')
            for i in range(len(self.data.profile)):
                out.write('# Profile ' + str(i + 1) + ' at ' + sf0(self.data.profile[i][0]) + '\n')
                out.write('# stagger angle = ' + sf0(self.data.profile[i][1].stagger_ang) +
                          ',   inlet angle = ' + sf0(self.data.profile[i][1].beta_1k) +
                          ',   outlet angle = ' + sf0(self.data.profile[i][1].beta_2k) + '\n')
                for j in range(self.data.profile[i][1].profile.shape[0]):
                    out.write(sf(self.data.profile[i][1].profile[i, 0]) + '    ' +
                              sf(self.data.profile[i][1].profile[i, 1]) + '    ' +
                              sf(self.data.profile[i][1].profile[i, 2]) + '    ' + '\n')
            out.close()
        print('output OVER')


# 叶片厚度分布类
class ThickDistribution:

    def __init__(self, geom_para, clx, gr_para):
        # arg为分布点数或中弧线
        self.para = geom_para
        self.npt = gr_para['num_cl']
        self.clx = clx
        self.npt = clx.shape[0]
        self.Thick = np.zeros([self.npt, 2], dtype=float, order='C')
        self.Thick[:, 0] = clx
        if gr_para['td_mtd'] == 'curva control':
            ThickDistribution.generate_by_curva(self)
        elif gr_para['td_mtd'] == 'B spline':
            ThickDistribution.generate_by_Bspline(self, geom_para['s3'], geom_para['T_tan_vec0'],
                                                  geom_para['T_tan_vec1'], geom_para['T_curva0'], geom_para['T_curva1'])
        elif gr_para['td_mtd'] == 'cubic polynomial':
            ThickDistribution.generate_by_cubic_poly(self)
        else:
            print("暂不支持其他厚度分布方法，请用3次多项式方法 'cubic polynomial'")
            sys.exit()

    def generate_by_curva(self):
        curva_ctr_pt_new = np.zeros((9, 2), dtype=float, order='C')
        c_ctr_pts = self.para['c_ctr_pts']
        s_ctr_pts = self.para['s_ctr_pts']
        curva_ctr_pt_new = np.array([[1.15924854e-01, -5.16720708e-06],
                                     [s_ctr_pts[0], c_ctr_pts[0]],
                                     [s_ctr_pts[1], c_ctr_pts[1]],
                                     [s_ctr_pts[2], c_ctr_pts[2]],
                                     [s_ctr_pts[3], c_ctr_pts[3]],
                                     [s_ctr_pts[4], c_ctr_pts[4]],
                                     [s_ctr_pts[5], c_ctr_pts[5]],
                                     [s_ctr_pts[6], c_ctr_pts[6]],
                                     [4.40216252e+01, -1.59860356e-02]])
        T_curva_new = cm.BSplineCurv(self.npt, curva_ctr_pt_new, 3)
        T_curva_new.generate_curve()
        T_tan_vec = T_curva_new.return_tangent_vector()
        T_curva_new = T_curva_new.curv
        T_vec_new = scint.cumtrapz(T_curva_new[:, 1], T_curva_new[:, 0], initial=0.)
        T_vec_new = T_vec_new * 1.178 + 1.6368
        T_tan_vec0 = [T_tan_vec[0, 0], T_vec_new[0]]
        T_tan_vec0 = geom.normalization(T_tan_vec0)
        tempk = np.sign(T_vec_new[-1]) * (T_vec_new[-1] ** 2 / (1 - T_vec_new[-1] ** 2)) ** 0.5
        T_tan_vec1 = [1 / (1 + tempk ** 2) ** 0.5, tempk / (1 + tempk ** 2) ** 0.5]
        T_tan_vec1 = geom.normalization(T_tan_vec1)

        # 积分, G(0)=r1, G(1)=r2
        # integrals = []
        # for i in range(len(y)): # 计算梯形的面积，由于是累加，所以是切片"i+1"
        #    integrals.append(scint.trapz(y[:i + 1], x[:i + 1]))
        ctr_pt_T = np.zeros([7, 2], dtype=float, order='C')
        s0 = self.clx[0]
        s6 = self.clx[-1]
        s1 = 2 * s0
        s2 = 2.8 * s0
        s5 = s6 - 3.3 * s0
        s4 = s5 - 7 * s0
        ctr_pt_T[0, :] = [s0, self.para['r1']]
        ctr_pt_T[1, :] = [s1, ctr_pt_T[0, 1] + T_tan_vec0[1] / T_tan_vec0[0] * (s1 - s0)]
        ctr_pt_T[2, :] = [s2, ctr_pt_T[1, 1] + T_tan_vec0[0] ** 2 * T_curva_new[0, 1] / 48 + T_tan_vec0[1] / T_tan_vec0[
            0] * (s2 - s1)]
        ctr_pt_T[3, :] = [self.para['thick_max_posi'], self.para['max_thick']]
        ctr_pt_T[6, :] = [s6, self.para['r2']]
        ctr_pt_T[5, :] = [s5, ctr_pt_T[6, 1] - T_tan_vec1[1] / T_tan_vec1[0] * (s6 - s5)]
        ctr_pt_T[4, :] = [s4,
                          ctr_pt_T[5, 1] + T_tan_vec1[0] ** 2 * T_curva_new[-1, 1] / 48 + T_tan_vec1[1] / T_tan_vec1[
                              0] * (s4 - s5)]
        self.Thick = cm.BSplineCurv(self.npt, ctr_pt_T, 3)
        self.Thick.generate_curve()
        self.T = self.Thick.curv
        print(self.T[np.argmax(self.T[:, 1]), :])
        # print(ctr_pt_T)
        # print('curva generation OVER')
        # print(T_tan_vec0, T_tan_vec1)
        # plt.scatter(curva_ctr_pt_new[:, 0], curva_ctr_pt_new[:, 1], label='Control points for curvature distribution')
        # plt.plot(T_curva_new[:, 0], T_curva_new[:, 1], label='Curvature distribution')
        tempcurva = np.abs(T_curva_new[:, 1])
        fig, ax1 = plt.subplots(1, 1)
        x1 = self.T[:, 0]
        y1 = self.T[:, 1]
        ax1.plot(x1, y1, label='Thickness distribution')

        ax2 = ax1.twinx()
        x2 = T_curva_new[:, 0]
        y2 = tempcurva
        ax2.plot(x2, y2, label='Curvature Distribution', color='orange')
        ax1.set_xlabel("s/mm")
        ax1.set_ylabel('h/mm')
        ax2.set_ylabel('Curvature')
        # plt.plot(T_curva_new[:, 0], tempcurva, label='Curvature Distribution')
        plt.scatter(s_ctr_pts, np.abs(c_ctr_pts), label='Control Points', color='black')
        plt.plot(s_ctr_pts, np.abs(c_ctr_pts), color='black', linestyle='--')
        # plt.plot(self.T[:, 0], self.T[:, 1], label = 'Thickness distribution')

    def generate_by_cubic_poly(self):
        a3 = sympy.symbols('a3')
        a2 = sympy.symbols('a2')
        a1 = sympy.symbols('a1')
        a0 = sympy.symbols('a0')
        solution = sympy.solve(
            [a0 - self.para.r1,
             a3 * self.para.b ** 3 + a2 * self.para.b ** 2 + a1 * self.para.b + a0 - self.para.r2,
             a3 * self.para.e ** 3 + a2 * self.para.e ** 2 + a1 * self.para.e + a0 - self.para.C_max,
             3. * a3 * self.para.e ** 2 + 2. * a2 * self.para.e + a1], [a0, a1, a2, a3])
        coef = np.array(solution)
        x = self.Thick[:, 0]
        self.Thick[:, 1] = coef[3] * x ** 3 + coef[2] * x ** 2 + coef[1] * x + coef[0]

    def generate_by_Bspline(self, s3, T_tan_vec0, T_tan_vec1, T_curva0, T_curva1):
        npt = self.npt
        ctr_pt_T = np.zeros([7, 2], dtype=float, order='C')
        s0 = self.clx[0]
        s6 = self.clx[-1]
        s1 = 2 * s0
        s2 = 2.8 * s0
        s5 = s6 - 3.3 * s0
        s4 = s5 - 7 * s0
        ctr_pt_T[0, :] = [s0, self.para['r1']]
        ctr_pt_T[1, :] = [s1, ctr_pt_T[0, 1] + T_tan_vec0[1] / T_tan_vec0[0] * (s1 - s0)]
        ctr_pt_T[2, :] = [s2, ctr_pt_T[1, 1] + T_tan_vec0[0] ** 2 * T_curva0 / 48 + T_tan_vec0[1] / T_tan_vec0[0] * (
                    s2 - s1)]
        ctr_pt_T[3, :] = [s3, self.para['max_thick']]
        ctr_pt_T[6, :] = [s6, self.para['r2']]
        ctr_pt_T[5, :] = [s5, ctr_pt_T[6, 1] - T_tan_vec1[1] / T_tan_vec1[0] * (s6 - s5)]
        ctr_pt_T[4, :] = [s4, ctr_pt_T[5, 1] + T_tan_vec1[0] ** 2 * T_curva1 / 48 + T_tan_vec1[1] / T_tan_vec1[0] * (
                    s4 - s5)]

        """ctr_pt1_new = cm.BezierCurv(self.npt, ctr_pt1).order_elevation(1)
        ctr_pt2_new = cm.BezierCurv(self.npt, ctr_pt2).order_elevation(1)"""

        print(ctr_pt_T)
        self.Thick = cm.BSplineCurv(npt, ctr_pt_T, 3)
        self.Thick.generate_curve()
        T_tan_vec = self.Thick.return_tangent_vector()
        T_curva = self.Thick.return_curvature()
        self.T = self.Thick.curv

        print('curva generation OVER')
        print(T_tan_vec0, T_tan_vec1)
        print(T_curva[0], T_curva[-1])

        # plt.plot(self.T[:, 0], self.T[:, 1], label = 'Thickness distribution')
        # plt.plot(self.T[:, 0], T_tan_vec[:, 1], label = 'Thickness distribution tan', linestyle = '--')
        # plt.plot(self.T[:, 0], T_curva, label = 'Thickness distribution curva', linestyle = '--')
        # plt.axis('equal')
        # plt.ylim(-2,3)
        # plt.legend()
        # plt.draw()
        # plt.figure(4)
        # plt.plot(curvePt[:, 0], curvePt[:, 1], label = 'Thickness distribution')
        # plt.plot(self.T[:, 0], T_curva, label = 'Thickness distribution curva', linestyle = '--')
        # plt.legend()
        # plt.draw()
