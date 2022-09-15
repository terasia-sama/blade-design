import sys
import numpy as np
import geometric as geom
import curve_modeling as cm
from blade import MainProfile


def generate_ss(gr_para, ss_ctr_pt, type='modify', *begin_info):
    """para: 生成参数"""
    '''type区分修型还是造型，默认修型'''
    npt = gr_para['num_ss']
    # 区分修型还是造型
    if type == 'modify':
        mtd = gr_para['ss_mtd']
    elif type == 'model':
        mtd = gr_para['ss_model_mtd']

    if mtd == 'cubic_spline':
        ss = cm.CubicSplineCurv(npt, ss_ctr_pt)
    elif mtd == 'Bezier':
        ss = cm.BezierCurv(npt, ss_ctr_pt)
    elif mtd == 'B-spline':
        ss = cm.BSplineCurv(gr_para['num_ss'], ss_ctr_pt, gr_para['ss_order'])
        mtd = gr_para['ss_mtd'] + ' ' + str(gr_para['ss_order']) + 'th'
    elif mtd == 'direct_profile':
        ss = DPCurveSS(npt, ss_ctr_pt, *begin_info, **argdict)
        #le.generate(af_para['le_curva'])
        mtd = gr_para['le_model_mtd']

    else:
        print('暂时木有这种吸力面曲线的拟合方法，请检查是否有拼写错误或者换个生成方法')
        sys.exit()

    ss.generate_curve()
    ss_tan_vec = ss.return_tangent_vector()
    ss_curva = ss.return_curvature()
    # print('ss_generation OVER')
    return ss.curv, ss_tan_vec, ss_curva, mtd


#新加入，直接型线法四点三阶吸力面，仍需调试
class DPCurveSS:

    def __init__(self, npt, ss_ctr_pt, *begin_info):
        """
        npt: 生成前缘上分布点数
        airfoil: 翼型类，含ss和ps两型面点集的坐标信息，需要已知ss ps与前缘衔接点的信息
        end_tan_vec: ss和ps首端的切矢组成的元组，若不输入则由坐标差分得到
        argdict: 参数字典，key包含'cl1', 'cl1_vec'
        """
        self.nss = npt/3    #分三段生成吸力面
        #pt1
        self.ss1 = airfoil.ss[0, :]
        self.ps1 = airfoil.ps[0, :]
        #k1,curva1,还需补充k0 curva0
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
        #
        self.le_pt = airfoil.le_pt
        self.chi1 = airfoil.para.chi1


    def generate(self, curva0):
        print('le_generation Begin')
        pt1 = geom.rotate_points(self.ss1 - self.le_pt, -self.chi1, np.array([0.0, 0.0]))
        temp_vec1 = geom.rotate_points(self.ss1_vec, -self.chi1, np.array([0.0, 0.0]))
        k0 = np.inf
        k1 = temp_vec1[1] / temp_vec1[0]
        nle_ss = int(self.nle/2)

        curva0 = -4
        print('curva0 = ',curva0)


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


def generate_ps(gr_para, ps_ctr_pt,type='modify',*begin_info):
    """para: 生成参数"""
    npt = gr_para['num_ps']
    # 区分修型还是造型
    if type == 'modify':
        mtd = gr_para['ss_mtd']
    elif type == 'model':
        mtd = gr_para['ss_model_mtd']

    if gr_para['ps_mtd'] == 'cubic_spline':
        ps = cm.CubicSplineCurv(npt, ps_ctr_pt)
    elif gr_para['ps_mtd'] == 'Bezier':
        ps = cm.BezierCurv(npt, ps_ctr_pt)
    elif gr_para['ps_mtd'] == 'B-spline':
        ps = cm.BSplineCurv(gr_para['num_ps'], ps_ctr_pt, gr_para['ps_order'])
        mtd = gr_para['ps_mtd'] + ' ' + str(gr_para['ps_order']) + 'th'
    elif mtd == 'direct_profile':
        ss = DPCurveSS(npt, airfoil, *begin_info, **argdict)
        #le.generate(af_para['le_curva'])
        mtd = gr_para['le_model_mtd']
    else:
        print('暂时木有这种吸力面曲线的拟合方法，请检查是否有拼写错误或者换个生成方法')
        sys.exit()

    ps.generate_curve()
    ps_tan_vec = ps.return_tangent_vector()
    ps_curva = ps.return_curvature()
    # print('ps_generation OVER')
    return ps.curv, ps_tan_vec, ps_curva, mtd

#新加入，直接型线法四点三阶压力面，仍需调试
class DPCurvePS:

    def __init__(self, npt, airfoil, *begin_info):
        """
        npt: 生成前缘上分布点数
        airfoil: 翼型类，含ss和ps两型面点集的坐标信息，需要已知ss ps与前缘衔接点的信息
        end_tan_vec: ss和ps首端的切矢组成的元组，若不输入则由坐标差分得到
        argdict: 参数字典，key包含'cl1', 'cl1_vec'
        """
        self.nss = npt/3    #分三段生成吸力面
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
        #
        self.le_pt = airfoil.le_pt
        self.chi1 = airfoil.para.chi1

    def generate(self, curva0):
        print('le_generation Begin')
        pt1 = geom.rotate_points(self.ss1 - self.le_pt, -self.chi1, np.array([0.0, 0.0]))
        temp_vec1 = geom.rotate_points(self.ss1_vec, -self.chi1, np.array([0.0, 0.0]))
        k0 = np.inf
        k1 = temp_vec1[1] / temp_vec1[0]
        nle_ss = int(self.nle/2)
        print('curva0 = ',curva0)
        '''
        if curva0 == -3:
            xm = 0.47
        elif curva0 == -3.5:
            xm = 0.21
        elif curva0 == -4:
            xm = 0.1
        else:
            xm = 0
        '''
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