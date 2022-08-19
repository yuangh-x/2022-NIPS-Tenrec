import random
import pandas as pd
from tqdm import tqdm

source_path = 'data/sbr_data_1M.csv'
df = pd.read_csv(source_path)
user_df = df.groupby('user_id')
columns = df.columns
del df
tmp_list = []
i = 0
for k, v in tqdm(user_df):
    tmp_list.extend(v.values.tolist())
    i += 1
    if i == 500000:
        part = pd.DataFrame(tmp_list, columns=columns)
        part.to_csv('data/task_0.csv', header=True, index=False)
        break

target_path1 = 'data/QK-article.csv'
target_path2 = 'data/QB-video.csv'
target_path3 = 'data/QB-article.csv'

source_df = part['user_id']
source_set = set(source_df.user_id.values.to_list())
target_df1 = pd.read_csv(target_path1)
target_set1 = set(target_df1.user_id.values.to_list())
overlap_set1 = source_set & target_set1
overlap_data1 = target_df1[target_df1.user_id.isin(list(overlap_set1))]

overlap_data1.to_csv('data/task_1.csv', header=True, index=False) 

target_df2 = pd.read_csv(target_path2)
target_set2 = set(target_df2.user_id.values.to_list())
overlap_set2 = source_set & target_set2
overlap_data2 = target_df2[target_df2.user_id.isin(list(overlap_set2))]

overlap_data2.to_csv('data/task_2.csv', header=True, index=False)

target_df3 = pd.read_csv(target_path3)
target_set3 = set(target_df3.user_id.values.to_list())
overlap_set3 = source_set & target_set3
overlap_data3 = target_df1[target_df3.user_id.isin(list(overlap_set3))]

overlap_data1.to_csv('data/task_3.csv', header=True, index=False)