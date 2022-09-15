# coding:utf-8

from airfoil_modeling import *
from airfoil_optimize import *
import numpy as np
import geometric as geom
import curve_modeling as cm
import time
from optimize import *
import os
import pandas as pd
import matplotlib.pyplot as plt
from cfx_automation import ANSYSCalling, call_multi_cfx_solve



# 基本变量
npt = 150
num_camber = 200
num_le = 101
num_te = 51
num_ss = 200
num_ps = 200

order = 3
stagger_angle = 41.91543286259973 / 180 * np.pi


# 翼型相关变量
b = 60  # 叶片弦长
r1 = 0.197
#以厚度分布与前、尾缘衔接点曲率为优化变量需要更改以下内容
# r2 = 0.683, 0.599
r2 = 0.470
chi1 = 59.61 / 180 * np.pi  # 几何进气角
chi2 = 32.77 / 180 * np.pi  # 几何出气角
center_le = [2.03191953e-01, 5.28068310e-02]  # 前缘中心位置
center_te = [5.92096677e+01, 5.71254762e-02]  # 尾缘中心位置
max_fle = 3.449287  # 最大挠度
fle_max_posi = 23.508145
max_thick = 2.377073902557191
thick_max_posi = 19.17642347 * np.cos(stagger_angle) - 3.35355428 * np.sin(stagger_angle)
psi1 = 0.823075
psi2 = 0.111682

center_le = [2.03191953e-01, 5.28068310e-02]  # 前缘中心位置
center_te = [5.92096677e+01, 5.71254762e-02]  # 尾缘中心位置

d_LE = len_LE = d_TE = len_TE =0.
le_curva = 0.
C_max=0.

s_ctr_pts = np.zeros((7, 1), dtype=float, order='C')
s_ctr_pts = [3.73812028e-01, 6.85009986e-01, 2.19010766e+00, 4.67203531e+00, 4.08371736e+01, 4.36134030e+01, 4.38286122e+01]
c_ctr_pts = np.zeros((7, 1), dtype=float, order='C')
c_ctr_pts = [-1.91556932, -0.25972606, -0.04500207,  0.0196401,  -0.01542469, -0.03284997, -1.11461655]


# generation parameter
GE_ctr_para = {'cl_mtd': 'parametric_design', 'td_mtd': 'curva control', 'le_mtd': 'arc', 'te_mtd': 'arc',
               'ss_mtd': 'B-spline','ps_mtd': 'B-spline',
               # 以下参数为新增
               'le_model_mtd': 'direct_profile', #适用于需要同时进行修型与造型的场合，否则与前面的方法应当设置为一致
               'te_model_mtd': 'direct_profile',
               'ss_model_mtd': 'B-spline',
               'ps_model_mtd': 'B-spline',
               'le_order': order, 'te_order': order,
               'cl_order': order,
               'ss_order': order,
               'ps_order': order,
               'input_path': 'D:/SJTU/Master22/Blade design/input/NACA_4digit.txt',
               'num_ss': num_ss,
               'num_ps': num_ps,
               #
               'num_le': num_le, 'num_te': num_te, 'num_cl': num_camber}

cl_ctr_pts = np.array([])
ss1_tan_vec = ss1_curva = ps1_tan_vec = ps1_curva = ss1 = ps1 = np.array([])

# 叶片参数变量
AF_ctr_para = {'chord': b, 'C_max': C_max, 'thick_max_posi': thick_max_posi, 'max_fle': max_fle,
               'fle_max_posi': fle_max_posi, 'stagger_angle': stagger_angle, 's_ctr_pts': s_ctr_pts,'c_ctr_pts': c_ctr_pts,
               #以下参数为新增
               #'ss_ctr_pts': ss_ctr_pts,
               'cl_ctr_pts': cl_ctr_pts,
               'cl_order': order,
               #'cl_order': cl_order,
               'd_LE': d_LE,
               'len_LE': len_LE,
               'd_TE': d_TE,
               'len_TE': len_TE,
               'le_curva':le_curva,#前缘点曲率
               'ss1_tan_vec':ss1_tan_vec,
               'ss1_curva':ss1_curva,
               'ps1_tan_vec':ps1_tan_vec,
               'ps1_curva':ps1_curva,
               'ss1':ss1,
               'ps1':ps1,

               #
               'r1': r1, 'r2': r2, 'psi1': psi1, 'psi2': psi2, 'chi1': chi1, 'chi2': chi2, 'center_le': center_le,
               'center_te': center_te}

edge_scope = {'le_ss': 10,
              'le_ps': 10,
              'te_ss': 10,
              'te_ps': 10,

}

# 修型
amodify = AirfoilModification(GE_ctr_para, AF_ctr_para, edge_scope)
[dic_update, gen_cl, gen_cl_tan_vec] = amodify.modify
AF_ctr_para.update(dic_update)
amodify.plot()

'''
# 中弧线控制点，从camber_line.py移出
[max_fle, fle_max_posi, chi1, chi2] = [AF_ctr_para['max_fle'], AF_ctr_para['fle_max_posi'], AF_ctr_para['chi1'], AF_ctr_para['chi2']]
A = [11.70452905, 3.10396417]
B = [34.57835527, 3.25155637]
factor = np.array([5.50883156, 2.12958718, 0.92718746, 1.62355577])
cl_ctr_pts = np.array([center_le,
                      [center_le[0] + (max_fle - center_le[1]) / np.tan(chi1) *
                      factor[2], max_fle / factor[0]],
                      A,
                      [fle_max_posi * 0.97, max_fle * 1.035],
                      B,
                      [center_te[0] + (center_te[1] - max_fle) / np.tan(chi2) *
                      factor[3], max_fle / factor[1]],
                      center_te])
AF_ctr_para['cl_ctr_pts'] = cl_ctr_pts
'''

am = AirfoilModeling(GE_ctr_para, AF_ctr_para)
am.para_design_mtd()
#am.direct_profile_mtd()
am.plot()


plt.show()
