import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from keras.models import Sequential
from keras.layers import Dense
from datetime import datetime
from sklearn.metrics import r2_score
import time

# 记录程序开始时间
start_time = time.time()

# 读取数据
data = pd.read_excel('D:/Shuidong/PCA_output/8input8output-afterGRA-pca-forLSTM.xlsx', header=0, index_col=0)
X = data.iloc[:, 1].values.reshape(-1, 1)
y = data.iloc[:, 2].values.reshape(-1, 1)

# 划分训练集和测试集
train_size = int(len(X) * 0.1)
train_indices = np.random.choice(len(X), size=train_size, replace=False)
test_indices = np.array(list(set(range(len(X))) - set(train_indices)))
X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

# 数据预处理
n_features = X.shape[1]  # 特征数量

# 构建BPNN模型
model = Sequential()
model.add(Dense(50, activation='relu', input_dim=n_features))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# 训练模型
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=2)

# 测试模型
y_pred = model.predict(X_test)
mse = np.mean((y_test - y_pred) ** 2)
r2 = r2_score(y_test, y_pred)

print('MSE: ', mse)
print('R2 Score: ', r2)

# 计算并输出程序运行时间
end_time = time.time()
total_time = end_time - start_time
print(f'Total runtime: {total_time:.2f} seconds')


# 可视化结果
fig, ax = plt.subplots(figsize=(15, 10))
ax.scatter(data.index[test_indices], y_test, s=10, label='Actual')
ax.scatter(data.index[test_indices], y_pred, s=10, label='Predicted')
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.rcParams['legend.fontsize'] = 15
ax.set_title('BPNN Model Prediction', fontsize=16)  # 设置标题字体大小为16

# 设置X轴和Y轴标签字体大小
ax.set_xlabel('Date', fontsize=15)  # 设置X轴标签字体大小为12
ax.set_ylabel('Seepage Volume', fontsize=15)  # 设置Y轴标签字体大小为12

# 设置刻度标签字体大小
ax.tick_params(axis='x', labelsize=13)  # 设置X轴刻度标签字体大小为10
ax.tick_params(axis='y', labelsize=15)  # 设置Y轴刻度标签字体大小为10

plt.legend()

# 输出结果到文件
now = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f'D:/Shuidong/LSTM_output/8input8output-afterGRA-pca-forLSTM_{now}.xlsx'

result_df = pd.DataFrame({'Actual': y_test.ravel(), 'Predicted': y_pred.ravel(), 'Label': data.index[test_indices]})
result_df.to_excel(filename, index=False)

plt.show()
