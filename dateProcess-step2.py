import pandas as pd

# 读取Excel文件
df = pd.read_excel('d:/shuidong/original_output/merged_data_total.xlsx')

# 提取日期部分作为新的一列
df['date'] = df[df.columns[0]].dt.date

# 按照日期进行分组，将其他列的值合并为逗号分隔的字符串，并且将有值的列求平均
df_grouped = df.groupby('date').agg(lambda x: ','.join(map(str, x.drop_duplicates().dropna().values)) if x.notna().any() else '')

# 重置索引
df_grouped = df_grouped.reset_index()

# 将平均值合并到结果中
for column in df.columns[1:]:
    if df[column].dtype == 'float64' or df[column].dtype == 'int64':
        df_grouped[column] = df.groupby('date')[column].mean().values

# 将结果写入Excel文件
df_grouped.to_excel('d:/shuidong/original_output/merge-data-total-grouped.xlsx', index=False)