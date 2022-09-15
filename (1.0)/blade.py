import numpy as np
import geometric as geom
import sys
import curve_modeling as cm

# 定义了许多基础类

## 基础翼型
# 中弧线类
class CambLine:

    def __init__(self, *chord_len):     #*parameter是用来接受任意多个参数并将其放在一个元组中
        """
        b:弦长 chord length;    a:最大挠度位置;    f_max:最大挠度;
        chi1:叶型前缘角 angle of LE;    chi2:叶型尾缘角 angle of TE;    theta:叶型弯角 camber angle;
        """
        if chord_len:
            self.b = chord_len[0]
        else:
            self.b = 1.0

        self.a = 0
        self.f_max = 0
        # 相对长度量
        self._a = 0
        self._f_max = 0
        # 角度量
        self.chi1 = 0
        self.chi2 = 0
        self.theta = 0
        self.curv = np.array([])

    def calc_relative_val(self):
        self._a = self.a / self.b
        self._f_max = self.f_max / self.b

    def calc_theta(self):
        self.theta = self.chi1 + self.chi2

    def calc_para_by_curv(self):
        self.b = np.linalg.norm(self.curv[-1, :] - self.curv[0, :])
        #计算F-范数,参考 https://blog.csdn.net/bitcarmanlee/article/details/51945271
        self.a = self.curv[np.argmax(self.curv[:, 1]), 0]
        self.f_max = np.max(self.curv[:, 1])
        CambLine.calc_relative_val(self)
        tan_vec1 = self.curv[1, :] - self.curv[0, :]
        tan_vec2 = self.curv[-1, :] - self.curv[-2, :]
        self.chi1 = abs(np.arctan(tan_vec1[1] / tan_vec1[0]) / np.pi * 180)
        self.chi2 = abs(np.arctan(tan_vec2[1] / tan_vec2[0]) / np.pi * 180)
        CambLine.calc_theta(self)

# 新添加
# 主型线类
# 需修改，吸力面的参数化方法
class MainProfile:

    def __init__(self, *chord_len):     #*parameter是用来接受任意多个参数并将其放在一个元组中
        """
        b:弦长 chord length;       f_max:最大挠度;    pos_f=[a,fa]:最大挠度的弦向、挠度位置;
        c_max:最大厚度;      pos_c=[e,fe]最大厚度的弦向、挠度位置
        """
        if chord_len:
            self.b = chord_len[0]
        else:
            self.b = 1.0

        self.f_max = 0
        self.a = 0
        self.fa = self.f_max
        self.pos_f=[a,fa]

        self.c_max = 0
        self.e = 0
        self.fe = 0
        self.pos_c = [e, fe]

        # 相对长度量
        self._a = 0
        self._f_max = 0
        self._fa = self._f_max

        self._e = 0
        self._fe = 0
        self._c_max = 0

    def calc_relative_val(self):
        self._a = self.a / self.b
        self._f_max = self.f_max / self.b
        self._c_max = self.c_max / self.b

'''
    def calc_para_by_curv(self):
        self.b = np.linalg.norm(self.curv[-1, :] - self.curv[0, :])
        #计算F-范数,参考 https://blog.csdn.net/bitcarmanlee/article/details/51945271
        self.a = self.curv[np.argmax(self.curv[:, 1]), 0]
        self.f_max = np.max(self.curv[:, 1])
        CambLine.calc_relative_val(self)
        tan_vec1 = self.curv[1, :] - self.curv[0, :]
        tan_vec2 = self.curv[-1, :] - self.curv[-2, :]
        self.chi1 = abs(np.arctan(tan_vec1[1] / tan_vec1[0]) / np.pi * 180)
        self.chi2 = abs(np.arctan(tan_vec2[1] / tan_vec2[0]) / np.pi * 180)
        CambLine.calc_theta(self)
'''

# 前缘类
class LeadingEdge:

    def __init__(self):
        self.le_ss = np.array([])
        self.le_ps = np.array([])
        self.chi = 0
        self.r = 0
        self.psi = 0
        self.curva = 0


# 翼型几何参数
class AirfoilPara:

    def __init__(self, *chord_len):
        """
        b:弦长 chord length;    a:最大挠度位置;    e:最大厚度位置;    f_max:最大挠度;    C_max最大厚度;
        r1:前缘小圆半径 LE circle radius;    r2:尾缘小圆半径 TE circle radius;
        chi1:叶型前缘角 angle of LE;    chi2:叶型尾缘角 angle of TE;    theta:叶型弯角 camber angle;
        psi1:前缘楔角 LE wedge angle;    psi2:尾缘楔角 TE wedge angle.
        """
        # 长度量
        if chord_len:
            self.b = chord_len[0]
        else:
            self.b = 1.0
        self.a = 0
        self.e = 0
        self.f_max = 0
        self.C_max = 0
        self.r1 = 0
        self.r2 = 0
        # 相对长度量
        self._a = 0
        self._e = 0
        self._f_max = 0
        self._C_max = 0
        self._r1 = 0
        self._r2 = 0
        # 角度量
        self.chi1 = 0
        self.chi2 = 0
        self.theta = 0
        self.psi1 = 0
        self.psi2 = 0

        #前缘点
        self.le_pt = np.array([])

    def calc_relative_val(self):
        self._a = self.a / self.b
        self._e = self.e / self.b
        self._f_max = self.f_max / self.b
        self._C_max = self.C_max / self.b
        self._r1 = self.r1 / self.b
        self._r2 = self.r2 / self.b

    def calc_theta(self):
        self.theta = self.chi1 + self.chi2


# 2维平面叶栅几何参数，继承自翼型几何参数AirfoilPara类
class CascadePara(AirfoilPara):

    def __init__(self, *chord_len):
        """
        stagger_ang:叶型安装角;   space:栅距;   solidity:稠度;   beta_1k:几何进口角;   beta_2k:几何出口角
        """
        if chord_len:
            super(CascadePara, self).__init__(chord_len)
        else:
            super(CascadePara, self).__init__()
        self.stagger_ang = 0
        self.space = 0
        self.solidity = 0
        self.beta_1k = 0
        self.beta_2k = 0

    def calc_solidity(self):
        self.solidity = self.b / self.space

    def calc_geom_inout_angle(self):
        self.beta_1k = self.stagger_ang - self.chi1
        self.beta_2k = self.stagger_ang + self.chi2


# 翼型：参数与型线
class Airfoil:

    def __init__(self, *geom_para):
        """
        geom_para: 几何参数，AirfoilPara类
        其他的型线点： nparray类型的 nx2 二维数组
        le和te首层为字典，分别包含其ss和ps两段型线，均以前/尾缘点为起始
        """
        if geom_para:
            self.para = geom_para
        else:
            self.para = AirfoilPara()
        self.cl = np.array([])
        self.ss = np.array([])
        self.ps = np.array([])
        self.le = {'le_ss': np.array([]), 'le_ps': np.array([]), 'le_int': np.array([])}
        self.te = {'te_ss': np.array([]), 'te_ps': np.array([]), 'te_int': np.array([])}
        self.int_ss = np.array([])
        self.int_ps = np.array([])
        self.profile = np.array([])

    # 对前缘处理，self.le['le_ss']去掉第一项，倒排之后，在添加上self.le['le_ps']
    def assemble_int_le(self):
        """从吸力面到压力面"""
        self.le['le_int'] = np.append(np.delete(self.le['le_ss'], 0, axis=0)[::-1], self.le['le_ps'], axis=0)


    # 对尾缘处理
    def assemble_int_te(self):
        """从吸力面到压力面"""
        self.te['te_int'] = np.append(np.delete(self.te['te_ss'], 0, axis=0)[::-1], self.te['te_ps'], axis=0)


    # 形成吸力面参数化曲线，吸力面前缘+叶身段+吸力面尾缘倒排
    def assemble_int_ss(self):
        """从前缘点到尾缘点"""
        self.int_ss = np.append(self.le['le_ss'], self.ss, axis=0)
        self.int_ss = np.append(self.int_ss, self.te['te_ss'][::-1, :], axis=0)

    # 形成压力面参数化曲线
    def assemble_int_ps(self):
        """从前缘点到尾缘点"""
        self.int_ps = np.append(self.le['le_ps'], self.ps, axis=0)
        self.int_ps = np.append(self.int_ps, self.te['te_ps'][::-1, :], axis=0)

    # 形成整个叶型参数化曲线
    def assemble_profile(self):
        """从前缘点开始，向ss方向排列，首尾点不重合"""
        self.profile = np.append(self.le['le_ss'], self.ss, axis=0)
        self.profile = np.append(self.profile, self.te['te_ss'][::-1, :], axis=0)
        self.profile = np.append(self.profile, np.delete(self.te['te_ps'], 0, axis=0), axis=0)
        self.profile = np.append(self.profile, self.ps[::-1, :], axis=0)
        self.profile = np.append(self.profile, np.delete(self.le['le_ps'], 0, axis=0)[::-1, :], axis=0)

## 单个叶轮
# 2维平面叶栅：参数与型线，继承自Airfoil类
class Cascade(Airfoil):

    def __init__(self, *geom_para):
        """
        geom_para: 几何参数，CascadePara类
        """
        super(Cascade, self).__init__(geom_para)    #继承自Airfoil类的初始化函数
        if geom_para:
            self.para = geom_para
        else:
            self.para = CascadePara()

    #能够直接用Cascade类调用相关参数，而不用再调用一次Airfoil类
    def inherit_airfoil_data(self, airfoil):
        self.a = airfoil.para.a
        self.e = airfoil.para.e
        self.f_max = airfoil.para.f_max
        self.C_max = airfoil.para.C_max
        self.r1 = airfoil.para.r1
        self.r2 = airfoil.para.r2
        # 相对长度量
        self._a = airfoil.para._a
        self._e = airfoil.para._e
        self._f_max = airfoil.para._f_max
        self._C_max = airfoil.para._C_max
        self._r1 = airfoil.para._r1
        self._r2 = airfoil.para._r2
        # 角度量
        self.chi1 = airfoil.para.chi1
        self.chi2 = airfoil.para.chi2
        self.theta = airfoil.para.theta
        self.psi1 = airfoil.para.psi1
        self.psi2 = airfoil.para.psi2
        # 点数据
        self.cl = airfoil.cl
        self.ss = airfoil.ss
        self.ps = airfoil.ps
        self.le = airfoil.le
        self.te = airfoil.te
        self.int_ss = airfoil.int_ss
        self.int_ps = airfoil.int_ps
        self.profile = airfoil.profile

    def install(self, direction, base_pt):
        if direction == 'CCW':
            alpha = self.para.stagger_ang
        elif direction == 'CW':
            alpha = -self.para.stagger_ang
        else:
            print("输入的方向参数必须是字符串'CW'（顺时针）或'CCW'（逆时针），请检查！")
            sys.exit()
        self.cl = geom.rotate_points(self.cl, alpha, base_pt)
        """将点（集）绕某点pt0逆时针旋转alpha (rad)角度  return: 点（集）"""
        self.ss = geom.rotate_points(self.ss, alpha, base_pt)
        self.ps = geom.rotate_points(self.ps, alpha, base_pt)
        self.le['le_ss'] = geom.rotate_points(self.le['le_ss'], alpha, base_pt)
        self.le['le_ps'] = geom.rotate_points(self.le['le_ps'], alpha, base_pt)
        self.te['te_ss'] = geom.rotate_points(self.te['te_ss'], alpha, base_pt)
        self.te['te_ps'] = geom.rotate_points(self.te['te_ps'], alpha, base_pt)
        if self.le['le_int']:
            self.le['le_int'] = geom.rotate_points(self.le['le_int'], alpha, base_pt)
        if self.te['te_int']:
            self.te['te_int'] = geom.rotate_points(self.te['te_int'], alpha, base_pt)
        if self.int_ss:
            self.int_ss = geom.rotate_points(self.int_ss, alpha, base_pt)
        if self.int_ps:
            self.int_ps = geom.rotate_points(self.int_ps, alpha, base_pt)
        if self.profile:
            self.profile = geom.rotate_points(self.profile, alpha, base_pt)


# 3维叶片：沿叶高方向由多个2维平面叶栅组成
class Blade:

    def __init__(self, span, n_section):
        self.span = span
        self.n_sec = n_section
        self.profile = [[row / (self.n_sec - 1), Cascade()] for row in range(self.n_sec)]
        self.hub = 0
        self.shroud = 0
        self.mid_r = 0
        self.hub_ratio = 1

    def calc_hub_tip(self):
        if self.mid_r == 0:
            print("50%叶高半径=0，默认为平面叶栅，轮毂比为默认值1")
        else:
            self.hub_ratio = (self.mid_r - self.span/2) / (self.mid_r + self.span/2)


## 一级叶轮
class Stage:

    def __init__(self, n_blade):
        self.kind = 'S'  # 'R'  'S'  'IGV'
        self.num = n_blade
        self.rpm = 0