import pickle
import joblib
import random
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
# from sklearn.preprocessing import LabelEncoder

source_path = 'data/QK-video.csv'
target_path = 'data/QK-article.csv'

source_df = pd.read_csv(source_path, usecols=['user_id'])
source_set = set(source_df.user_id.values.to_list())
target_df = pd.read_csv(target_path)
target_set = set(target_df.user_id.values.to_list())
overlap_set = source_set & target_set
overlap_data = target_df[target_df.user_id.isin(list(overlap_set))]

df = overlap_data

df = df[df.click.isin([1])]
new_list = []
user_counts = df.groupby('user_id').size()
user_set = set(df['user_id'].values.tolist())
cold_rate = 0.3 #0.7, 1
cold_num = int(len(user_set) * cold_rate)
cold_set = random.sample(list(user_set), cold_num)
for user in tqdm(set(df['user_id'].values.tolist())):
    if user in cold_set:
        rand = random.randint(1, 5)
        user_list = df[df['user_id'] == user].iloc[:rand, :].values.tolist()
    else:
        user_list = df[df['user_id'] == user].values.tolist()
    new_list.extend(user_list)
    # new_df.append(user_df)
new_df = pd.DataFrame(new_list, columns=df.columns)
new_df.to_csv('data/cold_data_{}.csv'.format(cold_rate), header=True, index=False)

new_list1 = []
for user in tqdm(set(df['user_id'].values.tolist())):
    user_list = df[df['user_id'] == user].iloc[:5, :].values.tolist()
    new_list1.extend(user_list)
new_df1 = pd.DataFrame(new_list1, columns=df.columns)
new_df1.to_csv('data/cold_data.csv', header=True, index=False)
