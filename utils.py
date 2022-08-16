import torch
import json
import joblib
import pickle
import torch.utils.data as data_utils
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import log_loss, roc_auc_score
from torch.utils.data.distributed import DistributedSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from model.ctr.inputs import *

tqdm.pandas()

def mtl_data(path=None, args=None):
    if not path:
        return
    df = pd.read_csv(path, usecols=["user_id", "item_id", "click", "exp", "like", "short_v", "gender", "age", "hist_1", "hist_2",
                       "hist_3", "hist_4", "hist_5", "hist_6", "hist_7", "hist_8", "hist_9", "hist_10"])
    # df = df[:100000]
    df['short_v'] = df['short_v'].astype(str)
    df = sample_data(df)
    if args.mtl_task_num == 2:
        label_columns = ['click', 'like']
        categorical_columns = ["user_id", "item_id", "short_v", "gender", "age", "hist_1", "hist_2",
                       "hist_3", "hist_4", "hist_5", "hist_6", "hist_7", "hist_8", "hist_9", "hist_10"]
    elif args.mtl_task_num == 1:
        label_columns = ['click']
        categorical_columns = ["user_id", "item_id", "short_v", "gender", "age", "hist_1", "hist_2",
                               "hist_3", "hist_4", "hist_5", "hist_6", "hist_7", "hist_8", "hist_9", "hist_10"]
    else:
        label_columns = ['like']
        categorical_columns = ["user_id", "item_id", "short_v", "gender", "age", "hist_1", "hist_2",
                               "hist_3", "hist_4", "hist_5", "hist_6", "hist_7", "hist_8", "hist_9", "hist_10"]
    user_columns = ["user_id", "gender", "age"]
    for col in tqdm(categorical_columns):
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    new_columns = categorical_columns + label_columns
    df = df.reindex(columns=new_columns)

    user_feature_dict, item_feature_dict = {}, {}
    for idx, col in tqdm(enumerate(df.columns)):
        if col not in label_columns:
            if col in user_columns:
                user_feature_dict[col] = (len(df[col].unique()), idx)
            else:
                item_feature_dict[col] = (len(df[col].unique()), idx)

    df = df.sample(frac=1)
    train_len = int(len(df) * 0.8)
    train_df = df[:train_len]
    tmp_df = df[train_len:]
    val_df = tmp_df[:int(len(tmp_df)/2)]
    test_df = tmp_df[int(len(tmp_df)/2):]
    return train_df, val_df, test_df, user_feature_dict, item_feature_dict

def set_fenbu(row_data):
    tmp1 = row_data[row_data.click.isin([0])]
    tmp2 = row_data[row_data.click.isin([1])]
    data = []
    j = 0
    for i in tqdm(range(int(len(tmp2)/1000))):
        data.append(tmp2.iloc[i, :].values.tolist())
        data.extend(tmp1.iloc[j : j + 3, :].values.tolist())
        j = j + 3
    new_data = pd.DataFrame(data, columns=row_data.columns)
    return new_data

def sample_data(df):
    p_df = df[df.click.isin([1])]
    n_df = df[df.click.isin([0]) & df.exp.isin([1])]
    del df
    n_df = n_df.sample(n=len(p_df)*2)
    df = p_df.append(n_df)
    del p_df, n_df
    df = df.sample(frac=1)
    return df

def ctr_share_dataset(path=None):
    if not path:
        return
    df = pd.read_csv(path, usecols=["user_id", "item_id", "click", "video_category", "gender", "age", "hist_1", "hist_2",
                       "hist_3", "hist_4", "hist_5", "hist_6", "hist_7", "hist_8", "hist_9", "hist_10"])
    df['video_category'] = df['video_category'].astype(str)
    df = sample_data(df)

    sparse_features = ["user_id", "item_id", "video_category", "gender", "age"]
    hist_feature = ["hist_1", "hist_2",
                       "hist_3", "hist_4", "hist_5", "hist_6", "hist_7", "hist_8", "hist_9", "hist_10"]
    lbe = LabelEncoder()
    df['click'] = lbe.fit_transform(df['click'])

    hist_enc = LabelEncoder()
    for feat in tqdm(sparse_features + hist_feature): #
        if 'item_id' in feat or 'hist' in feat:
            df[feat] = hist_enc.fit_transform(df[feat])
        else:
            lbe = LabelEncoder()
            df[feat] = lbe.fit_transform(df[feat])

    hist_set = set(df['hist_1'])
    hist_set.update(df['hist_2'])
    hist_set.update(df['hist_3'])
    hist_set.update(df['hist_4'])
    hist_set.update(df['hist_5'])
    hist_set.update(df['hist_6'])
    hist_set.update(df['hist_7'])
    hist_set.update(df['hist_8'])
    hist_set.update(df['hist_9'])
    hist_set.update(df['hist_10'])
    hist_set.update(df['item_id'])

    # hist concat
    hist_df = df[["user_id", "hist_1", "hist_2", "hist_3", "hist_4", "hist_5", "hist_6", "hist_7", "hist_8", "hist_9",
                  "hist_10"]]
    hist_df = hist_df.drop_duplicates()
    hist_dict = hist_df.set_index('user_id').T.to_dict('list')

    df['hist_item_id'] = df['user_id']
    df['hist_item_id'] = df['hist_item_id'].map(hist_dict)

    # df['hist_item_id'] = np.ndarray(df['hist_item_id'])
    df.drop(columns=["hist_1", "hist_2", "hist_3", "hist_4", "hist_5", "hist_6", "hist_7", "hist_8", "hist_9", "hist_10"], inplace=True)
    df['seq_length'] = 10

    fixlen_feature_columns = []
    for feat in sparse_features:
        if feat == 'item_id':
            fixlen_feature_columns.append(SparseFeat(feat, len(hist_set)))
        else:
            fixlen_feature_columns.append(SparseFeat(feat, df[feat].nunique()))

    fixlen_feature_columns += [
        VarLenSparseFeat(
            SparseFeat('hist_item_id', vocabulary_size=len(hist_set), embedding_dim=32),
            maxlen=10, length_name="seq_length")]
    linear_feature_columns = fixlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
    train, test = train_test_split(df, test_size=0.1)
    del df
    train_hist_item = np.array(train['hist_item_id'].values.tolist())
    test_hist_item = np.array(test['hist_item_id'].values.tolist())

    train_model_input = {name: train[name] for name in feature_names}
    train_model_input['hist_item_id'] = train_hist_item
    test_model_input = {name: test[name] for name in feature_names}
    test_model_input['hist_item_id'] = test_hist_item
    return train, test, train_model_input, test_model_input, linear_feature_columns, dnn_feature_columns

def ctr_din_dataset(path=None):
    if not path:
        return
    df = pd.read_csv(path, usecols=["user_id", "item_id", "click", "video_category", "gender", "age", "hist_1", "hist_2",
                       "hist_3", "hist_4", "hist_5", "hist_6", "hist_7", "hist_8", "hist_9", "hist_10"])
    df['video_category'] = df['video_category'].astype(str)

    df = sample_data(df)

    sparse_features = ["user_id", "item_id", "video_category", "gender", "age"]
    hist_feature = ["hist_1", "hist_2",
                       "hist_3", "hist_4", "hist_5", "hist_6", "hist_7", "hist_8", "hist_9", "hist_10"]
    lbe = LabelEncoder()
    df['click'] = lbe.fit_transform(df['click'])
    #
    hist_enc = LabelEncoder()
    for feat in tqdm(sparse_features + hist_feature):
        if 'item_id' in feat or 'hist' in feat:
            df[feat] = hist_enc.fit_transform(df[feat])
        else:
            lbe = LabelEncoder()
            df[feat] = lbe.fit_transform(df[feat])

    hist_set = set(df['hist_1'])
    hist_set.update(df['hist_2'])
    hist_set.update(df['hist_3'])
    hist_set.update(df['hist_4'])
    hist_set.update(df['hist_5'])
    hist_set.update(df['hist_6'])
    hist_set.update(df['hist_7'])
    hist_set.update(df['hist_8'])
    hist_set.update(df['hist_9'])
    hist_set.update(df['hist_10'])
    hist_set.update(df['item_id'])

    # hist concat
    hist_df = df[["user_id", "hist_1", "hist_2", "hist_3", "hist_4", "hist_5", "hist_6", "hist_7", "hist_8", "hist_9",
                  "hist_10"]]
    hist_df = hist_df.drop_duplicates()
    hist_dict = hist_df.set_index('user_id').T.to_dict('list')

    df['hist_item_id'] = df['user_id']
    df['hist_item_id'] = df['hist_item_id'].map(hist_dict)

    df.drop(columns=["hist_1", "hist_2", "hist_3", "hist_4", "hist_5", "hist_6", "hist_7", "hist_8", "hist_9", "hist_10"], inplace=True)
    df['seq_length'] = 10

    fixlen_feature_columns = []
    for feat in sparse_features:
        if feat == 'item_id':
            fixlen_feature_columns.append(SparseFeat(feat, len(hist_set)))
        else:
            fixlen_feature_columns.append(SparseFeat(feat, df[feat].nunique()))


    hist_list = ['item_id']

    fixlen_feature_columns += [
        VarLenSparseFeat(
            SparseFeat('hist_item_id', vocabulary_size=len(hist_set), embedding_dim=32),
            maxlen=10, length_name="seq_length")]
    linear_feature_columns = fixlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
    train, test = train_test_split(df, test_size=0.1)
    del df
    train_hist_item = np.array(train['hist_item_id'].values.tolist())
    test_hist_item = np.array(test['hist_item_id'].values.tolist())
    train_model_input = {name: train[name] for name in feature_names}
    train_model_input['hist_item_id'] = train_hist_item
    test_model_input = {name: test[name] for name in feature_names}
    test_model_input['hist_item_id'] = test_hist_item
    return train, test, train_model_input, test_model_input, dnn_feature_columns, hist_list

def ctrdataset(path=None):
    if not path:
        return
    df = pd.read_csv(path, usecols=["user_id", "item_id", "click", "exp", "short_v", "gender", "age", "hist_1", "hist_2",
                       "hist_3", "hist_4", "hist_5", "hist_6", "hist_7", "hist_8", "hist_9", "hist_10"])
    df['short_v'] = df['short_v'].astype(str)
    df = sample_data(df)
    sparse_features = ["user_id", "item_id", "short_v", "gender", "age", "hist_1", "hist_2",
                       "hist_3", "hist_4", "hist_5", "hist_6", "hist_7", "hist_8", "hist_9", "hist_10"]

    lbe = LabelEncoder()
    df['click'] = lbe.fit_transform(df['click'])

    for feat in tqdm(sparse_features):
        lbe = LabelEncoder()
        df[feat] = lbe.fit_transform(df[feat])
    fixlen_feature_columns = [SparseFeat(feat, df[feat].nunique())
                              for feat in sparse_features]
    linear_feature_columns = fixlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
    train, test = train_test_split(df, test_size=0.1)
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}
    return train, test, train_model_input, test_model_input, linear_feature_columns, dnn_feature_columns

def split_warm_hot(df, item_min):
    user_counts = df.groupby('user_id').size()
    w_user_subset = np.in1d(df.user_id, user_counts[user_counts >= item_min].index)
    c_user_subset = np.in1d(df.user_id, user_counts[(user_counts <= 5) & (user_counts > 1)].index)
    w_filter_df = df[w_user_subset].reset_index(drop=True)
    c_filter_df = df[c_user_subset].reset_index(drop=True)
    return w_filter_df, c_filter_df

def transferdataset(args):
    path = args.path
    item_min = args.item_min
    if not path:
        return
    df = pd.read_csv(path, usecols=['user_id', 'item_id', 'click'])

    df = df[df.click.isin([1])]
    user_counts = df.groupby('user_id').size()
    user_subset = np.in1d(df.user_id, user_counts[user_counts >= item_min].index)
    filter_df = df[user_subset].reset_index(drop=True)

    user_count = len(set(filter_df['user_id']))
    item_count = len(set(filter_df['item_id']))

    assert (filter_df.groupby('user_id').size() < item_min).sum() == 0
    del df

    reset_ob = reset_df()
    filter_df = reset_ob.fit_transform(filter_df)

    user_history = {}
    print("+++user_history+++")
    savefile_path = Path("data/history_{}.pkl".format(args.seed))
    if savefile_path.is_file():
        with open(savefile_path, "rb") as load_f:
            user_history = pickle.load(load_f)
    else:
        for uid in tqdm(filter_df.user_id.unique()):
            dataframe = filter_df[filter_df.user_id == uid].item_id
            sequence = dataframe.values.tolist()
            user_history[uid] = sequence
        with open(savefile_path, "wb") as dump_f:
            pickle.dump(user_history, dump_f)
    return filter_df, user_history, user_count, item_count

def sequencedataset(item_min, args, path=None):
    if '2' in path:
        df = pd.read_csv(path, usecols=['user_id', 'item_id', 'like'])
        df = df[df.like.isin([1])]
    else:
        df = pd.read_csv(path, usecols=['user_id', 'item_id', 'click'])
        df = df[df.click.isin([1])]
    user_counts = df.groupby('user_id').size()
    user_subset = np.in1d(df.user_id, user_counts[user_counts >= item_min].index)
    filter_df = df[user_subset].reset_index(drop=True)

    assert (filter_df.groupby('user_id').size() < item_min).sum() == 0
    user_count = len(set(filter_df['user_id']))
    item_count = len(set(filter_df['item_id']))
    del df

    reset_ob = reset_df()
    filter_df = reset_ob.fit_transform(filter_df)
    print("+++user_history+++")
    user_history = filter_df.groupby('user_id').item_id.apply(list).to_dict()
    return filter_df, user_history, user_count, item_count

def data_count(df, item_min, target=False):
    user_counts = df.groupby('user_id').size()
    if target:
        filter_df = df
    else:
        user_subset = np.in1d(df.user_id, user_counts[user_counts >= item_min].index)
        filter_df = df[user_subset].reset_index(drop=True)

        assert (filter_df.groupby('user_id').size() < item_min).sum() == 0
    user_count = len(set(filter_df['user_id']))
    item_count = len(set(filter_df['item_id']))
    return filter_df, user_count, item_count


def construct_data(args, item_min):
    path1 = args.target_path
    path2 = args.source_path
    if args.task != 2:
        df1 = pd.read_csv(path1, usecols=['user_id', 'item_id', 'click'])
        df1 = df1[df1.click.isin([1])]
    else:
        df1 = pd.read_csv(path1, usecols=['user_id', 'item_id', 'like'])
        df1 = df1[df1.like.isin([1])]
    df2 = pd.read_csv(path2, usecols=['user_id', 'item_id', 'click'])
    df2 = df2[df2.click.isin([1])]
    user_counts = df2.groupby('user_id').size()
    user_subset = np.in1d(df2.user_id, user_counts[user_counts >= item_min].index)
    df2 = df2[user_subset].reset_index(drop=True)

    assert (df2.groupby('user_id').size() < item_min).sum() == 0
    s_item_count = len(set(df2['item_id']))
    reset_ob = cold_reset_df()
    df2, df1 = reset_ob.fit_transform(df2, df1)
    user1 = set(df1.user_id.values.tolist())
    user2 = set(df2.user_id.values.tolist())
    user = user1 & user2
    df1 = df1[df1.user_id.isin(list(user))]
    df2 = df2[df2.user_id.isin(list(user))]
    new_data1 = []
    new_data2 = []
    for u in user:
        tmp_data2 = df2[df2.user_id == u][:-3].values.tolist()
        if 'cold' in args.task_name:
            tmp_data1 = df1[df1.user_id == u].values.tolist()
        else:
            if args.task == 1:
                tmp_data1 = df1[df1.user_id == u][-3:].values.tolist()
            else:
                tmp_data1 = df1[df1.user_id == u].values.tolist()
        new_data1.extend(tmp_data1)
        new_data2.extend(tmp_data2)
    new_data1 = pd.DataFrame(new_data1, columns=df1.columns)
    new_data2 = pd.DataFrame(new_data2, columns=df2.columns)
    user_count = len(set(new_data1.user_id.values.tolist()))
    reset_item = item_reset_df()
    new_data1 = reset_item.fit_transform(new_data1)
    t_item_count = len(set(new_data1['item_id']))
    return new_data1, new_data2, user_count, t_item_count, s_item_count

def construct_ch_data(args, item_min):
    path1 = args.target_path
    path2 = args.source_path

    df1 = pd.read_csv(path1, usecols=['user_id', 'item_id', 'click'])
    df1 = df1[df1.click.isin([1])]
    df2 = pd.read_csv(path2, usecols=['user_id', 'item_id', 'click'])
    df2 = df2[df2.click.isin([1])]

    user_counts = df2.groupby('user_id').size()
    user_subset = np.in1d(df2.user_id, user_counts[user_counts >= item_min].index)
    df2 = df2[user_subset].reset_index(drop=True)

    assert (df2.groupby('user_id').size() < item_min).sum() == 0
    s_item_count = len(set(df2['item_id']))
    reset_ob = cold_reset_df()
    df2, df1 = reset_ob.fit_transform(df2, df1)

    user1 = set(df1.user_id.values.tolist())
    user2 = set(df2.user_id.values.tolist())
    user = user1 & user2
    # df = df[:100000]
    df1 = df1[df1.user_id.isin(list(user))]
    df2 = df2[df2.user_id.isin(list(user))]

    # cold and hot user
    user_counts1 = df1.groupby('user_id').size()
    cold_user_ind = np.in1d(df1.user_id, user_counts1[user_counts1 <= 5].index)
    hot_user_ind = np.in1d(df1.user_id, user_counts1[user_counts1 > 5].index)

    cold_user = set(df1[cold_user_ind].user_id.values.tolist())
    hot_user = set(df1[hot_user_ind].user_id.values.tolist())

    new_data1 = []
    new_data2 = []
    for u in user:
        tmp_data2 = df2[df2.user_id == u][:-3].values.tolist()
        tmp_data1 = df1[df1.user_id == u].values.tolist()
        new_data1.extend(tmp_data1)
        new_data2.extend(tmp_data2)
    new_data1 = pd.DataFrame(new_data1, columns=df1.columns)
    new_data2 = pd.DataFrame(new_data2, columns=df2.columns)
    user_count = len(set(new_data1.user_id.values.tolist()))
    reset_item = item_reset_df()
    new_data1 = reset_item.fit_transform(new_data1)
    t_item_count = len(set(new_data1['item_id']))
    return new_data1, new_data2, user_count, t_item_count, s_item_count, cold_user, hot_user

def colddataset(item_min, args, path=None):
    if args.ch:
        target_data, source_data, user_count, t_item_count, s_item_count, cold_user, hot_user = construct_ch_data(args, item_min)
    else:
        target_data, source_data, user_count, t_item_count, s_item_count = construct_data(args, item_min)
    print("+++user_history+++")
    user_history = source_data.groupby('user_id').item_id.apply(list).to_dict()
    target = target_data.groupby('user_id').item_id.apply(list).to_dict()

    if args.ch:
        hot_examples = []
        cold_examples = []
        for u, t_list in tqdm(target.items()):
            if u in cold_user:
                for t in t_list:
                    e_list = [user_history[u] + [0], t]
                    cold_examples.append(e_list)
            else:
                for t in t_list:
                    e_list = [user_history[u] + [0], t]
                    hot_examples.append(e_list)
        cold_examples = pd.DataFrame(cold_examples, columns=['source', 'target'])
        hot_examples = pd.DataFrame(hot_examples, columns=['source', 'target'])
        return cold_examples, hot_examples, user_count, s_item_count, t_item_count
    else:
        examples = []
        for u, t_list in tqdm(target.items()):
            for t in t_list:
                e_list = [user_history[u] + [0], t]
                examples.append(e_list)
        examples = pd.DataFrame(examples, columns=['source', 'target'])
        return examples, user_count, s_item_count, t_item_count

def lifelongdataset(item_min, args, path=None):
    target_data, source_data, user_count, t_item_count, s_item_count = construct_data(args, item_min)
    print("+++user_history+++")
    user_history = source_data.groupby('user_id').item_id.apply(list).to_dict()
    target = target_data.groupby('user_id').item_id.apply(list).to_dict()
    examples = []
    for u, t_list in tqdm(target.items()):
        for t in t_list:
            e_list = [user_history[u] + [0], t]
            examples.append(e_list)
    examples = pd.DataFrame(examples, columns=['source', 'target'])
    return examples, user_count, s_item_count, t_item_count

def profiledata(item_min, args, path=None):
    df = pd.read_csv(path, usecols=['user_id', 'item_id', 'click', 'gender', 'age'])
    df = df[df.click.isin([1])]  # [:500000]
    user_counts = df.groupby('user_id').size()
    user_subset = np.in1d(df.user_id, user_counts[user_counts >= item_min].index)
    filter_df = df[user_subset].reset_index(drop=True)

    assert (filter_df.groupby('user_id').size() < item_min).sum() == 0
    item_count = len(set(filter_df['item_id']))
    del df
    reset_ob = reset_df()
    df = reset_ob.fit_transform(filter_df)
    if args.user_profile == 'gender':
        df = df[df.gender != 0]

        df['gender'] = df['gender'] - 1
        user_count = len(set(df['user_id']))
        label_count = len(set(df['gender']))
        print("+++user_gender_dataframe+++")
        gender_list = []
        user_history = df.groupby('user_id').item_id.apply(list).to_dict()
        gender = df.groupby('user_id').gender.apply(list).to_dict()
        for u, his in user_history.items():
            tmp_list = [u, his[:-3], gender[u][0]]
            gender_list.append(tmp_list)
        profile_df = pd.DataFrame(gender_list, columns=['uid', 'history', 'profile'])
    elif args.user_profile == 'age':
        df = df[df.age != 0]
        df['age'] = df['age'] - 1
        user_count = len(set(df['user_id']))
        label_count = len(set(df['age']))
        print("+++user_age_dataframe+++")
        age_list = []
        user_history = df.groupby('user_id').item_id.apply(list).to_dict()
        age = df.groupby('user_id').age.apply(list).to_dict()
        for u, his in user_history.items():
            tmp_list = [u, his[:-3], age[u][0]]
            age_list.append(tmp_list)
        profile_df = pd.DataFrame(age_list, columns=['uid', 'history', 'profile'])
    return profile_df, user_count, item_count, label_count

def utils(df, args):
    if args.user_profile == 'gender':
        df = df[df.gender != 0]
        df = df[df.click.isin([1])]
        df['gender'] = df['gender'] - 1
        user_counts = df.groupby('user_id').size()
        user_subset = np.in1d(df.user_id, user_counts[user_counts >= args.item_min].index)
        filter_df = df[user_subset].reset_index(drop=True)
    elif args.user_profile == 'age':
        df = df[df.age != 0]
        df = df[df.click.isin([1])]
        df['age'] = df['age'] - 1
        user_counts = df.groupby('user_id').size()
        user_subset = np.in1d(df.user_id, user_counts[user_counts >= args.item_min].index)
        filter_df = df[user_subset].reset_index(drop=True)

    return filter_df

def gender_df(filter_df, args):
    if args.user_profile == 'gender':
        gender_list = []
        for uid in tqdm(filter_df.user_id.unique()):
            dataframe = filter_df[filter_df.user_id == uid].item_id
            sequence = dataframe.values.tolist()
            gender = filter_df[filter_df.user_id == uid].gender.values[0]
            list = [uid, sequence, gender]
            gender_list.append(list)
        profile_df = pd.DataFrame(gender_list, columns=['uid', 'history', 'profile'])
    else:
        age_list = []
        for uid in tqdm(filter_df.user_id.unique()):
            dataframe = filter_df[filter_df.user_id == uid].item_id
            sequence = dataframe.values.tolist()
            age = filter_df[filter_df.user_id == uid].age.values[0]
            list = [uid, sequence, age]
            age_list.append(list)
        profile_df = pd.DataFrame(age_list, columns=['uid', 'history', 'profile'])

    return profile_df

def train_val_test_split(user_history):
    if not user_history:
        return
    train_history = {}
    val_history = {}
    test_history = {}
    for key, history in tqdm(user_history.items()):
        train_history[key] = history[:-2]
        val_history[key] = history[-2:-1]
        test_history[key] = history[-1:]
    return train_history, val_history, test_history

def val_test_split(user_history):
    if not user_history:
        return
    val_history = {}
    test_history = {}
    val_len = int(len(user_history) / 5)
    test_len = int(len(user_history))
    i = 0
    for key, history in tqdm(user_history.items()):
        if i < val_len:
            val_history[key] = history
            i += 1
        elif i >= val_len and i < test_len:
            test_history[key] = history
            i += 1
    return val_history, test_history

class item_reset_df(object):

    def __init__(self):
        print("=" * 10, "Initialize Reset DataFrame Object", "=" * 10)
        self.item_enc = LabelEncoder()

    def fit_transform(self, df):
        print("=" * 10, "Resetting item ids in DataFrame", "=" * 10)
        df['item_id'] = self.item_enc.fit_transform(df['item_id']) + 1
        return df

    def inverse_transform(self, df):
        df['item_id'] = self.item_enc.inverse_transform(df['item_id']) - 1
        return df

class reset_df(object):

    def __init__(self):
        print("=" * 10, "Initialize Reset DataFrame Object", "=" * 10)
        self.item_enc = LabelEncoder()
        self.user_enc = LabelEncoder()

    def fit_transform(self, df):
        print("=" * 10, "Resetting user ids and item ids in DataFrame", "=" * 10)
        df['item_id'] = self.item_enc.fit_transform(df['item_id']) + 1
        df['user_id'] = self.user_enc.fit_transform(df['user_id']) + 1
        return df

    def inverse_transform(self, df):
        df['item_id'] = self.item_enc.inverse_transform(df['item_id']) - 1
        df['user_id'] = self.user_enc.inverse_transform(df['user_id']) - 1
        return df

class cold_reset_df(object):

    def __init__(self):
        print("=" * 10, "Initialize Reset DataFrame Object", "=" * 10)
        self.item_enc1 = LabelEncoder()
        self.item_enc2 = LabelEncoder()
        self.user_enc = LabelEncoder()

    def fit_transform(self, df1, df2):
        print("=" * 10, "Resetting user ids and item ids in DataFrame", "=" * 10)
        df = df1['user_id'].append(df2['user_id'])
        df = self.user_enc.fit_transform(df) + 1
        df1['item_id'] = self.item_enc1.fit_transform(df1['item_id']) + 1
        df1['user_id'] = df[:len(df1)]
        df2['item_id'] = self.item_enc2.fit_transform(df2['item_id']) + 1
        df2['user_id'] = df[len(df1):]
        return df1, df2

    def inverse_transform(self, df):
        df['item_id'] = self.item_enc.inverse_transform(df['item_id']) - 1
        df['user_id'] = self.user_enc.inverse_transform(df['user_id']) - 1
        return df

class mtlDataSet(data_utils.Dataset):
    def __init__(self, data, args):
        self.feature = data[0]
        self.args = args
        if args.mtl_task_num == 2:
            self.label1 = data[1]
            self.label2 = data[2]
        else:
            self.label = data[1]

    def __getitem__(self, index):
        feature = self.feature[index]
        if self.args.mtl_task_num == 2:
            label1 = self.label1[index]
            label2 = self.label2[index]
            return feature, label1, label2
        else:
            label = self.label[index]
            return feature, label

    def __len__(self):
        return len(self.feature)

class ProfileDataset(data_utils.Dataset):
    def __init__(self, x, y, max_len, mask_token):
        self.seqs = x
        self.targets = y
        self.max_len = max_len
        self.mask_token = mask_token

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        seq = self.seqs[index]
        target = self.targets[index]
        seq = seq + [self.mask_token]
        seq = seq[-self.max_len:]
        seq_len = len(seq)
        seq_mask_len = self.max_len - seq_len
        seq = [0] * seq_mask_len + seq
        return torch.LongTensor(seq), torch.LongTensor([target])

class BuildTrainDataset(data_utils.Dataset):
    def __init__(self, u2seq, max_len, mask_prob, mask_token, num_items, rng):#
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        self.num_items = num_items
        self.rng = rng

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self._getseq(user)

        tokens = seq[:-1]
        labels = seq[1:]
        #
        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]

        x_len = len(tokens)
        y_len = len(labels)

        x_mask_len = self.max_len - x_len
        y_mask_len = self.max_len - y_len


        tokens = [0] * x_mask_len + tokens
        labels = [0] * y_mask_len + labels

        return torch.LongTensor(tokens), torch.LongTensor(labels)

    def _getseq(self, user):
        return self.u2seq[user]

class ColdDataset(data_utils.Dataset):
    def __init__(self, x, y, max_len, mask_token):
        self.seqs = x
        self.targets = y
        self.max_len = max_len
        self.mask_token = mask_token

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        seq = self.seqs[index]
        target = self.targets[index]
        seq = seq[-self.max_len:]
        seq_len = len(seq)
        seq_mask_len = self.max_len - seq_len
        seq = [0] * seq_mask_len + seq
        return torch.LongTensor(seq), torch.LongTensor([target])

class ColdEvalDataset(data_utils.Dataset):
    def __init__(self, x, y, max_len, mask_token, num_item):
        self.seqs = x
        self.targets = y
        self.max_len = max_len
        self.mask_token = mask_token
        self.num_item = num_item + 1

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        seq = self.seqs[index]
        target = self.targets[index]
        labels = [0] * self.num_item
        labels[target] = 1
        seq = seq[-self.max_len:]
        seq_len = len(seq)
        seq_mask_len = self.max_len - seq_len
        seq = [self.mask_token] * seq_mask_len + seq
        return torch.LongTensor(seq), torch.LongTensor(labels)

class pos_neg_TrainDataset(data_utils.Dataset):
    def __init__(self, u2seq, max_len, mask_token, num_items):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.mask_token = mask_token
        self.num_items = num_items

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self._getseq(user)
        tokens = seq[:-1]
        pos = seq[1:]

        tokens = tokens[-self.max_len:]
        pos = pos[-self.max_len:]
        seen = set(tokens)
        seen.update(pos)
        neg = []
        for _ in range(len(pos)):
            item = np.random.choice(self.num_items + 1)  #
            while item in seen or item in neg:
                item = np.random.choice(self.num_items + 1)  #
            neg.append(item)

        neg = neg[-self.max_len:]

        x_len = len(tokens)
        p_len = len(pos)
        n_len = len(neg)

        x_mask_len = self.max_len - x_len
        p_mask_len = self.max_len - p_len
        n_mask_len = self.max_len - n_len

        tokens = [self.mask_token] * x_mask_len + tokens
        pos = [self.mask_token] * p_mask_len + pos
        neg = [self.mask_token] * n_mask_len + neg

        return torch.LongTensor(tokens), torch.LongTensor(pos), torch.LongTensor(neg)#, torch.LongTensor([x_len]), torch.LongTensor([user])

    def _getseq(self, user):
        return self.u2seq[user]

class BertTrainDataset(data_utils.Dataset):
    def __init__(self, u2seq, max_len, mask_prob, mask_token, num_items, rng):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        self.num_items = num_items
        self.rng = rng

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self._getseq(user)

        tokens = []
        labels = []
        for s in seq:
            prob = self.rng.random()
            if prob < self.mask_prob:
                prob /= self.mask_prob

                if prob < 0.8:
                    tokens.append(self.mask_token)
                elif prob < 0.9:
                    tokens.append(self.rng.randint(1, self.num_items))
                else:
                    tokens.append(s)

                labels.append(s)
            else:
                tokens.append(s)
                labels.append(0)

        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]

        mask_len = self.max_len - len(tokens)

        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels

        return torch.LongTensor(tokens), torch.LongTensor(labels)

    def _getseq(self, user):
        return self.u2seq[user]

class BuildEvalDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2answer, max_len, mask_token, negative_samples):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len
        self.mask_token = mask_token
        self.negative_samples = negative_samples

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]
        answer = self.u2answer[user]
        answer = answer[-1:]
        labels = answer
        seq = seq + [self.mask_token]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [self.mask_token] * padding_len + seq

        return torch.LongTensor(seq), torch.LongTensor(labels)

class Build_neg_EvalDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2answer, max_len, mask_token, item_count, neg_samples):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len
        self.mask_token = mask_token
        self.item_count = item_count
        self.neg_samples = neg_samples

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]
        answer = self.u2answer[user]
        answer = answer[-1:]
        negs = self.neg_samples[user]

        candidates = answer + negs
        labels = [1] * len(answer) + [0] * len(negs)
        seq = seq + [self.mask_token]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [self.mask_token] * padding_len + seq

        return torch.LongTensor(seq), torch.LongTensor(candidates), torch.LongTensor(labels)

class Build_full_EvalDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2answer, max_len, mask_token, num_item):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len
        self.mask_token = mask_token
        self.num_item = num_item + 1

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user][:-1]
        answer = self.u2answer[user]
        answer = answer[-1:][0]


        labels = [0] * self.num_item
        labels[answer] = 1
        seq = seq + [self.mask_token]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [self.mask_token] * padding_len + seq

        return torch.LongTensor(seq), torch.LongTensor(labels)

class new_Build_full_EvalDataset(data_utils.Dataset):
    def __init__(self, u2seq, max_len, mask_token, num_item):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        # self.u2answer = u2answer
        self.max_len = max_len
        self.mask_token = mask_token
        self.num_item = num_item

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user][:-2]
        answer = self.u2seq[user][-1:][0]

        labels = [0] * self.num_item
        labels[answer] = 1
        seq = seq + [self.mask_token]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [self.mask_token] * padding_len + seq

        return torch.LongTensor(seq), torch.LongTensor(labels)

def get_train_loader(dataset, args):
    if args.is_parallel:
        dataloader = data_utils.DataLoader(dataset, batch_size=args.train_batch_size, sampler=DistributedSampler(dataset))
    else:
        dataloader = data_utils.DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, pin_memory=True)
    return dataloader

def get_val_loader(dataset, args):
    if args.is_parallel:
        dataloader = data_utils.DataLoader(dataset, batch_size=args.val_batch_size, sampler=DistributedSampler(dataset))
    else:
        dataloader = data_utils.DataLoader(dataset, batch_size=args.val_batch_size, shuffle=False, pin_memory=True)
    return dataloader

def get_test_loader(dataset, args):
    if args.is_parallel:
        dataloader = data_utils.DataLoader(dataset, batch_size=args.test_batch_size, sampler=DistributedSampler(dataset))
    else:
        dataloader = data_utils.DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False, pin_memory=True)
    return dataloader

