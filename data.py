import numpy as np
import sys

# 标准 T
T_matrix = np.array([
    [1, 1, 1, 1, 1, 1],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
])

# 标准 J
J_matrix = np.array([
    [0, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 0, 0]
])

#增加噪声
def AddNoice(Matrix,NoiceLevel):
    noise = np.random.normal(0, NoiceLevel, (6,6))
    noisy_matrix = Matrix + noise
    return np.clip(noisy_matrix, 0, 1)

#输出每一个矩阵到CSV文件中
def Output_Csv(Matrix, FileName):
    np.savetxt('Matrix/'+str(FileName)+'.csv', Matrix, delimiter=',', fmt='%.3f')

for i in range(20):
    Output_Csv(AddNoice(T_matrix, 0.1), f'T_{i}')
    Output_Csv(AddNoice(J_matrix, 0.1), f'J_{i}')