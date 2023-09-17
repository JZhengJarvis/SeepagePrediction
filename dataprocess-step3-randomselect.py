import pandas as pd
import numpy as np

# 读取原始Excel文件
file_path = 'd:/shuidong/PCA_output/14-96-afterGRA-pca.xlsx'
df = pd.read_excel(file_path, header=0)

# 随机选取20%的行
sample_size = int(len(df) * 0.2)
sample_rows = np.random.choice(df.index, size=sample_size, replace=False)

# 将20%的行输出到新的Excel文件中
sample_df = df.loc[sample_rows]
sample_df.to_excel('d:/shuidong/SVR_input/sample_file.xlsx', index=False)

# 将剩余80%的行输出到另一个Excel文件中
rest_df = df.drop(sample_rows)
rest_df.to_excel('d:/shuidong/SVR_input/rest_file.xlsx', index=False)
# 读取训练数据
train_data = pd.read_excel('d:/shuidong/SVR_input/rest_file.xlsx')

# 分离X和y
y_train = train_data.iloc[:, 0].values
X_train = train_data.iloc[:, 1:].values

# 对X进行标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

