import os
import pandas as pd

# 输入和输出文件夹路径
input_folder = 'd:/shuidong/original_input'
output_folder = 'd:/shuidong/original_output'

# 获取输入文件夹内的所有 Excel 文件名
input_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.xlsx')]

# 读取所有 Excel 文件的数据到 pandas 数据框中
dfs = []
for f in input_files:
    print(f)
    df = pd.read_excel(f)
    dfs.append(df)

# 合并所有数据框到一个单独的数据框中
merged_df = pd.concat(dfs)

# 根据观测时间进行排序
merged_df.sort_values(by=['观测日期'], inplace=True)

# 将结果输出到指定的输出文件夹中
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

output_file = os.path.join(output_folder, 'merged_data_total.xlsx')
merged_df.to_excel(output_file, index=False)
