import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import os
import time
import seaborn as sns
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV

plot = 1
start_time = time.time()
# 读取原始Excel文件
file_path = 'd:/shuidong/PCA_output/8input8output-afterGRA-pca.xlsx'
df = pd.read_excel(file_path, header=0)

# 随机选取20%的行
sample_size = int(len(df) * 0.1)
sample_rows = np.random.choice(df.index, size=sample_size, replace=False)

# 将20%的行输出到新的Excel文件中
sample_df = df.loc[sample_rows]
sample_df.to_excel('d:/shuidong/SVR_input/sample_file.xlsx', index=False)

# 将剩余80%的行输出到另一个Excel文件中
rest_df = df.drop(sample_rows)
rest_df.to_excel('d:/shuidong/SVR_input/rest_file.xlsx', index=False)

# 读取训练数据
train_data = pd.read_excel('d:/shuidong/SVR_input/sample_file.xlsx', index_col=0)
train_data.index.name = 'label'

# 分离X和y
y_train = train_data.iloc[:, 0].values
X_train = train_data.iloc[:, 1:].values
print(X_train.shape)

# 对X进行标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# 设置SVR模型
svr = SVR(kernel='rbf')

param_grid = {'C': np.logspace(-2, 4, 7),
              'gamma': np.logspace(-3, 3, 7)}

Read_time = time.time()
print('Read time',Read_time-start_time)
# 使用GridSearchCV进行超参数搜索
grid = GridSearchCV(svr, param_grid, cv=5, n_jobs=-1, scoring='r2')
grid.fit(X_train, y_train)
# 输出最佳参数组合
print("Best parameters: ", grid.best_params_)
print("Best score: ", grid.best_score_)

# 提取网格搜索结果并生成DataFrame
results = pd.DataFrame(grid.cv_results_)
r2_scores = np.array(results.mean_test_score).reshape(len(param_grid['gamma']),len(param_grid['C']))

if plot:
    # 设置配色方案
    sns.set_palette('husl')

    # 设置字体
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12

    # 绘制R2的图表
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    param_C, param_gamma = np.meshgrid(param_grid['C'], param_grid['gamma'])
    ax.plot_surface(np.log10(param_gamma), np.log10(param_C), r2_scores, cmap=plt.cm.jet, alpha=0.8)
    ax.set_xlabel('log(C)')
    ax.set_ylabel('log(g)')
    ax.set_zlabel('$R^2$')
    ax.set_zticks(np.arange(0, 1, 0.2))
    #ax.set_title('Grid Search for R2')
    ax.view_init(elev=30, azim=135)
    plt.savefig('d:/shuidong/SVR_output/grid_search_r2.png', dpi=1200)
    plt.show()

Train_time = time.time()
print('Train',Train_time-Read_time)

# 使用最佳参数组合训练SVR模型并预测
best_svr = grid.best_estimator_

# 读取测试数据
test_data = pd.read_excel('d:/shuidong/SVR_input/rest_file.xlsx', index_col=0)
test_data.index.name = 'label'
test_data.columns = ['prediction'] + list(test_data.columns[1:])


# 分离X_test和y_test
y_test = test_data.iloc[:, 0].values
X_test = test_data.iloc[:, 1:].values

# 对X_test进行标准化
X_test = scaler.transform(X_test)

# 预测
y_pred = best_svr.predict(X_test)
Prediction_time = time.time()
print('Prediction time',Prediction_time-Train_time)
# 计算R2和MSE
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# 输出R2和MSE
print("R2: ", r2)
print("MSE: ", mse)
# 获取当前时间
now = datetime.datetime.now()


# 将预测结果，真实结果，预测精度，R2，MSE，最优超参数组合输出到excel中
result_df = pd.DataFrame({'Predicted': y_pred,
                          'Actual': y_test,
                          'Accuracy': (1 - abs(y_pred - y_test) / y_test) * 100,
                          'R2': r2,
                          'MSE': mse},
                          index=test_data.index)

# 获取当前时间
now = datetime.datetime.now()

# 将DataFrame输出到Excel文件
best_params_str = str(grid.best_params_).replace(':', '_').replace(',','_').replace('{','').replace('}','')
result_df.to_excel(f'd:/shuidong/SVR_output/result_{now.strftime("%Y%m%d%H%M%S")}_best{best_params_str}_input{os.path.splitext(os.path.basename(file_path))[0]}_time{Prediction_time-start_time}.xlsx')




# 画出预测值和真实值的图
if plot:
    plt.figure()
    plt.scatter(test_data.index, y_test, label='Actual',alpha=0.8,s=20,c='#1f77b4')
    plt.scatter(test_data.index, y_pred, label='Predicted',alpha=0.7,s=20,c='#e74c3c')
    plt.xlabel('Label')
    plt.ylabel('Value')
    plt.title('Actual vs Predicted Values')
    plt.legend()
    plt.savefig('d:/shuidong/SVR_output/prediction.png', dpi=1200)
    plt.show()
