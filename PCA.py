import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# 读取Excel文件
file_path = 'd:/shuidong/GRA_output/59-96.xlsx'
df = pd.read_excel(file_path, header=0)

# 将DataFrame转换为数组
X = df.values

# PCA降维
pca = PCA()
pca.fit(X)

# 计算累计贡献值
cumulative_contribution = np.cumsum(pca.explained_variance_ratio_)

# 指定降维后的维数K
K = 14

# 降维
pca = PCA(n_components=K)
pca.fit(X)
transformed_X = pca.transform(X)

# 将降维后的结果转换为DataFrame并输出到Excel文件
columns = ['PC{}'.format(i+1) for i in range(K)]
result_df = pd.DataFrame(transformed_X, columns=columns)
result_df.to_excel('d:/shuidong/PCA_output/59-96-afterGRA-pca-14.xlsx', index=False)

# 输出累计贡献值
for i, c in enumerate(cumulative_contribution[:K]):
    print('前{}个主成分的累计贡献值为{:.2%}'.format(i+1, c))
