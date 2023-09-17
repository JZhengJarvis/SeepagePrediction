import pandas as pd
import numpy as np

# 读取Excel文件
file_path = 'd:/shuidong/GRA_input/59-96.xlsx'
df = pd.read_excel(file_path, header=0)

# 分离第一列
y = df.iloc[:, 0].values

# 分离后面17列
x = df.iloc[:, 1:].values

# 计算灰色关联系数
def gm11(x0):
    n = len(x0)
    x1 = np.cumsum(x0)
    z1 = (x1[:-1] + x1[1:]) / 2.0
    z1 = z1.reshape((n - 1, 1))
    B = np.append(-z1, np.ones_like(z1), axis=1)
    Y = x0[1:].reshape((n - 1, 1))
    [[a], [b]] = np.dot(np.dot(np.linalg.inv(np.dot(B.T, B)), B.T), Y)
    result = (x0[0] - b / a) * np.exp(-a * np.arange(n))
    return result[-1]

# 计算灰色关联度
def grey_relation_degree(x0, y0):
    x_mean = np.mean(x0)
    y_mean = np.mean(y0)
    sxy = np.sum((x0 - x_mean) * (y0 - y_mean))
    sx2 = np.sum((x0 - x_mean) ** 2)
    sy2 = np.sum((y0 - y_mean) ** 2)
    if np.isnan(sx2) or np.isnan(sy2) or np.isnan(sxy):
        return 0
    else:
        r = sxy / np.sqrt(sx2 * sy2)
        return r


result = np.zeros(x.shape[1])

for i in range(x.shape[1]):
    result_i = gm11(x[:, i])
    if np.isnan(result_i):
        result[i] = np.nanmean(x[:, i])
    else:
        result[i] = result_i
    print(result[i],i)


# 计算灰色关联度并加入结果中
degree = np.zeros(x.shape[1])

for i in range(x.shape[1]):
    degree[i] = grey_relation_degree(x[:, i], y)

result_df = pd.DataFrame({'灰色关联系数': result, '关联度': degree}, index=df.columns[1:])
result_df.index.name = '因素'
result_df.to_excel('d:/shuidong/GRA_output/grey_relation_coefficients-59.xlsx')
