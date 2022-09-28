# 最小二乘法求解线性方程系数
# 系数求解结果：
# a = [0.06513968]
# b = [0.40962719]
# c = [0.17938605]
# d = [0.63365882]

import numpy as np
import pandas as pd

#构造方程 y = ax1+bx2+cx3+dx0
# A = [x11 x12 x13 x10
#      x21 x22 x23 x20
#      ....
#      xn1 xn2 xn3 xn0]
# b = [a b c d]T
# Y = [y1 y2 ... yn]T
# 由Ab = Y得(省略推导)：
# 系数项b = (AT * A)-1 *AT*Y
#
def LS_solve(data):
    col = data.shape[1]
    matA = np.mat(data.iloc[:,0:col-1])
    matY = np.mat(data.iloc[:,col-1]).T
    matATA = matA.T * matA
    invATA = matATA.I
    b = invATA * matA.T * matY
    return b;

#从CSV文件中读入数据
filepath = "data.csv"
df = pd.read_csv(filepath,sep = ",",names=['X1','X2','X3','Y'])
#构造常数特征项 X0 = ones(n,1)
X0 = [1 for index in range(len(df))]
df['X0'] = X0
df=df.reindex(columns=['X1','X2','X3','X0','Y'],fill_value=0)
#求解方程系数
b = LS_solve(df)

print(b)


