# import numpy as np
# import sympy as sp
# from scipy.optimize import root

# width = 30 / 100
# max_r = 4.5

# # 定义常量 a 的值
# a_val = 1.7 / 2 * np.pi

# EPLISON = 1e-5

# # 定义符号变量
# a, theta, theta1, theta2, R = sp.symbols("a theta theta1 theta2 R")

# # 阿基米德螺线方程
# x = a * theta * sp.cos(theta)
# y = a * theta * sp.sin(theta)

# # 阿基米德螺线的导数
# dx_dtheta = a * (sp.cos(theta) - theta * sp.sin(theta))
# dy_dtheta = a * (sp.sin(theta) + theta * sp.cos(theta))

# # 切线方向的法线方向（垂直）
# norm = sp.sqrt(dx_dtheta**2 + dy_dtheta**2)
# nx = -dy_dtheta / norm
# ny = dx_dtheta / norm

# # 圆心坐标
# x0 = x + R * nx
# y0 = y + R * ny

# # 将 theta 替换为 theta1 和 theta2
# r1 = R
# r2 = r1 / 2

# # 计算圆心坐标
# x1 = x0.subs(theta, theta1).subs(R, r1)
# y1 = y0.subs(theta, theta1).subs(R, r1)
# x2 = x0.subs(theta, theta2).subs(R, r2)
# y2 = y0.subs(theta, theta2).subs(R, r2)

# # 定义符号方程式
# eq = (x1 + x2) ** 2 + (y1 + y2) ** 2 - (3 * r2) ** 2

# # 将符号表达式转为数值函数，供 SciPy 使用
# f_numeric = sp.lambdify([theta1, theta2, a, R], eq, modules=["numpy"])

# # 生成一系列 theta1 值
# theta1_vals = np.linspace(2 * 2 * np.pi, (9 / 1.7) * np.pi, 100)
# r1_vals = np.linspace(width / 2, max_r, 100)

# # 用于存储 theta2 的解
# theta2_solutions = []


# # 定义目标函数，用于 root 求解
# def equation(theta2, theta1_val, r1_val):
#     return f_numeric(theta1_val, theta2, a_val, r1_val)


# # 遍历每个 theta1 和 r1 值，使用 root 求解对应的 theta2
# for theta1_val, r1_val in zip(theta1_vals, r1_vals):
#     initial_guess = 2.0  # 初始猜测值
#     while True:
#         try:
#             # 使用 root 函数求解
#             sol = root(
#                 equation,
#                 initial_guess,
#                 args=(theta1_val, r1_val),
#                 method="hybr",
#                 tol=1e-10,
#                 options={"maxfev": 10000},
#             )

#             # 检查解是否成功，并且 theta2 为正值
#             if sol.success and sol.x[0] > 0:
#                 theta2_solutions.append(sol.x[0])
#                 break  # 成功找到解，跳出循环
#             else:
#                 # 如果解失败或为负值，增加初始猜测值，重新求解
#                 initial_guess += 0.5

#         except Exception as e:
#             print(f"Failed to solve for theta1 = {theta1_val}: {e}")
#             theta2_solutions.append(None)
#             break

# # 输出 theta2 解
# print(theta2_solutions)

# # 将解带入方程，检查是否满足
# for theta1_val, r1_val, theta2_val in zip(theta1_vals, r1_vals, theta2_solutions):
#     if theta2_val is not None:
#         # 计算r2
#         r2_val = r2.subs(theta, theta1_val).subs(R, r1_val)
#         # 计算x1, y1, x2, y2
#         x1_val = x1.subs(theta1, theta1_val).subs(r1, r1_val).subs(a, a_val)
#         y1_val = y1.subs(theta1, theta1_val).subs(r1, r1_val).subs(a, a_val)
#         x2_val = x2.subs(theta2, theta2_val).subs(r1, r1_val).subs(a, a_val)
#         y2_val = y2.subs(theta2, theta2_val).subs(r1, r1_val).subs(a, a_val)
#         print(
#             f"theta1 = {theta1_val}, r1 = {r1_val}, theta2 = {theta2_val},r2 = {r2_val}"
#         )
#         print(f"x1 = {x1_val}, y1 = {y1_val}, x2 = {x2_val}, y2 = {y2_val}")
#         print(f"Equation value: {f_numeric(theta1_val, theta2_val, a_val, r1_val)}")
#         print("")
import random
import numpy as np
import sympy as sp
from scipy.optimize import minimize

width = 30 / 100
max_r = 4.5

# 定义常量 a 的值
a_val = 1.7 / 2 * np.pi

# 定义符号变量
a, theta, theta1, theta2, R = sp.symbols("a theta theta1 theta2 R")

# 阿基米德螺线方程
x = a * theta * sp.cos(theta)
y = a * theta * sp.sin(theta)

# 阿基米德螺线的导数
dx_dtheta = a * (sp.cos(theta) - theta * sp.sin(theta))
dy_dtheta = a * (sp.sin(theta) + theta * sp.cos(theta))

# 切线方向的法线方向（垂直）
norm = sp.sqrt(dx_dtheta**2 + dy_dtheta**2)
nx = -dy_dtheta / norm
ny = dx_dtheta / norm

# 圆心坐标
x0 = x + R * nx
y0 = y + R * ny

# 将 theta 替换为 theta1 和 theta2
r1 = R
r2 = r1 / 2

# 计算圆心坐标
x1 = x0.subs(theta, theta1).subs(R, r1)
y1 = y0.subs(theta, theta1).subs(R, r1)
x2 = x0.subs(theta, theta2).subs(R, r2)
y2 = y0.subs(theta, theta2).subs(R, r2)

# 定义符号方程式
eq = (x1 + x2) ** 2 + (y1 + y2) ** 2 - (3 * r2) ** 2

# 将符号表达式转为数值函数，供 SciPy 使用
f_numeric = sp.lambdify([theta1, theta2, a, R], eq, modules=["numpy"])

# 生成一系列 theta1 值
theta1_vals = np.linspace(2 * 2 * np.pi, (9 / 1.7) * np.pi, 100)
r1_vals = np.linspace(width / 2, max_r, 100)

# 用于存储 theta2 的解
theta2_solutions = []


# 定义目标函数，用于 minimize 求解
def equation(theta2, theta1_val, r1_val):
    return np.abs(f_numeric(theta1_val, theta2, a_val, r1_val))


# 遍历每个 theta1 和 r1 值，使用 minimize 求解对应的 theta2
for theta1_val, r1_val in zip(theta1_vals, r1_vals):
    initial_guess = random.uniform(0, (9 / 1.7) * np.pi)  # 初始猜测值随机扰动

    try:
        # 使用 minimize 函数求解，采用 BFGS 方法
        sol = minimize(
            equation,
            initial_guess,
            args=(theta1_val, r1_val),
            method="BFGS",
            tol=1e-5,
            options={"maxiter": 1000},  # 限制最大迭代次数，避免无效迭代
        )

        # 检查解是否成功，并且 theta2 为正值
        if sol.success and sol.x[0] > 0:
            theta2_solutions.append(sol.x[0])
        else:
            theta2_solutions.append(None)

    except Exception as e:
        print(f"Failed to solve for theta1 = {theta1_val}: {e}")
        theta2_solutions.append(None)

# 输出 theta2 解
print(theta2_solutions)

# 将解带入方程，检查是否满足
for theta1_val, r1_val, theta2_val in zip(theta1_vals, r1_vals, theta2_solutions):
    if theta2_val is not None:
        # 计算r2
        r2_val = r2.subs(theta, theta1_val).subs(R, r1_val)
        # 计算x1, y1, x2, y2
        x1_val = x1.subs(theta1, theta1_val).subs(R, r1_val).subs(a, a_val)
        y1_val = y1.subs(theta1, theta1_val).subs(R, r1_val).subs(a, a_val)
        x2_val = -x2.subs(theta2, theta2_val).subs(R, r1_val).subs(a, a_val)
        y2_val = -y2.subs(theta2, theta2_val).subs(R, r1_val).subs(a, a_val)
        print(
            f"theta1 = {theta1_val}, r1 = {r1_val}, theta2 = {theta2_val},r2 = {r2_val}"
        )
        print(f"x1 = {x1_val}, y1 = {y1_val}, x2 = {x2_val}, y2 = {y2_val}")
        print(f"Equation value: {f_numeric(theta1_val, theta2_val, a_val, r1_val)}")
        print("")
