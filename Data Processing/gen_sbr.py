import pandas as pd
from tqdm import tqdm

path = '/data/QK_video.csv'

source_data = pd.read_csv(path)
source_data = source_data[source_data.click.isin([1])]
soure_item = source_data['item_id'].value_counts()
soure_item = pd.DataFrame(soure_item)
high_item = soure_item[soure_item['item_id'] > 200000].index.to_list() #这里item_id代表频率
low_item = soure_item[soure_item['item_id'] < 2].index.to_list()

filter_item = high_item + low_item
new_data = source_data[~source_data['item_id'].isin(filter_item)]
# new_item = new_data['item_id'].value_counts()

df = new_data
user_dict = {}
user_df = df.groupby('user_id')
columns = df.columns
del df
tmp_list = []
i = 0
for k, v in tqdm(user_df):
    tmp_list.extend(v.values.tolist())
    i += 1
    if i == 1000000:
        part = pd.DataFrame(tmp_list, columns=columns)
        part.to_csv('/datat/sbr_data_1M.csv', header=True, index=False)
        break

