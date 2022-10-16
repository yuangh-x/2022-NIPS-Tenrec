import random
from abc import *
import numpy as np
from pathlib import Path
from collections import Counter
from tqdm import trange, tqdm
import pickle
from pathlib import Path

class AbstractNegativeSampler(metaclass=ABCMeta):
    def __init__(self, train, val, test, user_count, item_count, sample_size, seed, save_folder):
        self.train = train
        self.val = val
        self.test = test
        self.user_count = user_count
        self.item_count = item_count
        self.sample_size = sample_size
        self.seed = seed
        self.save_folder = save_folder

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def generate_negative_samples(self):
        pass

    def get_negative_samples(self):
        savefile_path = self._get_save_path()
        if savefile_path.is_file():
            print('Negatives samples exist. Loading.')
            negative_samples = pickle.load(savefile_path.open('rb'))
            return negative_samples
        print("Negative samples don't exist. Generating.")
        negative_samples = self.generate_negative_samples()
        with savefile_path.open('wb') as f:
            pickle.dump(negative_samples, f)
        return negative_samples

    def _get_save_path(self):
        folder = Path(self.save_folder)
        filename = '{}-sample_size{}-seed{}.pkl'.format(self.code(), self.sample_size, self.seed)
        return folder.joinpath(filename)

class RandomNegativeSampler(AbstractNegativeSampler):
    @classmethod
    def code(cls):
        return 'random'

    def generate_negative_samples(self):
        assert self.seed is not None, 'Specify seed for random sampling'
        np.random.seed(self.seed)
        negative_samples = {}
        # all_item = set(np.arange(1, self.user_count+1))
        print('Sampling negative items')
        for user in trange(1, self.user_count+1):#:self.train
        # for user in trange(0, self.user_count):  #:self.train
            if isinstance(self.train[user][1], tuple):
                seen = set(x[0] for x in self.train[user])
                seen.update(x[0] for x in self.val[user])
                seen.update(x[0] for x in self.test[user])
            else:
                seen = set(self.train[user])
                seen.update(self.val[user])
                seen.update(self.test[user])

            samples = []
            for _ in range(self.sample_size):
                item = np.random.choice(self.item_count + 1) #
                # item = np.random.choice(self.item_count)  #
                while item in seen or item in samples:
                    item = np.random.choice(self.item_count + 1) #
                    # item = np.random.choice(self.item_count)  #
                samples.append(item)
            # tmp_set = all_item - seen
            # samples = random.sample(list(tmp_set), self.sample_size)
            negative_samples[user] = samples

        return negative_samples

class PopularNegativeSampler(AbstractNegativeSampler):
    @classmethod
    def code(cls):
        return 'popular'

    def generate_negative_samples(self):
        popular_items_freq = self.items_by_popularity() #popular_items
        negative_samples = {}
        print('Sampling negative items')
        for user in trange(1, self.user_count+1):#+1
            seen = set(self.train[user])
            #evluate
            # seen.update(self.val[user])
            # seen.update(self.test[user])

            temp = popular_items_freq.copy()
            # samples = []
            for item in seen:
            # for item in popular_items_freq.keys():
                # if len(samples) == self.sample_size:
                #     break
                temp.pop(item)
                    # continue
                # samples.append(item)
            samples = random.choices(list(temp.keys()), weights=list(temp.values()), k=self.sample_size)
            negative_samples[user] = samples

        return negative_samples

    def items_by_popularity(self):
        popularity = Counter()
        for user in range(1, self.user_count+1):#(0, self.user_count):
            popularity.update(self.train[user])
            popularity.update(self.val[user])
            popularity.update(self.test[user])
        # popular_items = sorted(popularity, key=popularity.get, reverse=True)
        popular_items = dict(popularity.most_common(self.item_count))
        word_counts = np.array([count for count in popular_items.values()], dtype=np.float32)
        word_freqs = word_counts / np.sum(word_counts)
        word_freqs = word_freqs ** (3. / 4.)  # 论文中提到的将频率3/4 然后做归一化，对预测准确率有提高
        word_freqs = word_freqs / np.sum(word_freqs)
        i = 0
        for key, _ in popular_items.items():
            values = word_freqs[i]
            popular_items[key] = values
            i += 1
        return popular_items


def create_user_noclick(user_history, df, n_items, args):
    print("=" * 10, "Creating User 'no-click' history", "=" * 10)
    user_noclick = {}
    all_items = np.arange(n_items).tolist()

    item_counts = df.groupby('item_id', sort='item_id').size()
    # item_counts = (item_counts/item_counts.sum()).values
    save_path = Path('data/neg_history_{}.pkl'.format(args.seed))
    if save_path.is_file():
        with open(save_path, "rb") as load_f:
            user_noclick = pickle.load(load_f)
    else:
        for uid, history in tqdm(user_history.items()):
            # no_clicks = np.in1d(all_items, history)
            no_clicks = list(set(all_items) - set(history))#list(set.difference(set(all_items), set(history)))
            item_counts_subset = item_counts[no_clicks]
            probabilities = (item_counts_subset / item_counts_subset.sum()).values

            user_noclick[uid] = (no_clicks, probabilities)
        with open(save_path, "wb") as dump_f:
            pickle.dump(user_noclick, dump_f)

    return user_noclick


import numpy as np


class AbstractSampler(object):
    def __init__(self, args):
        self.uid_name = 'user_id'
        self.iid_name = 'item_id'
        self.item_num = args.num_items
        self.ur = args.train_ur

    def sampling(self):
        raise NotImplementedError


class BasicNegtiveSampler(AbstractSampler):
    def __init__(self, df, args):
        """
        negative sampling class for <u, pos_i, neg_i> or <u, pos_i, r>
        Parameters
        ----------
        df : pd.DataFrame, the raw <u, i, r> dataframe
        user_num: int, the number of users
        item_num: int, the number of items
        num_ng : int, No. of nagative sampling per sample, default is 4
        sample_method : str, sampling method, default is 'uniform',
                        'uniform' discrete uniform sampling
                        'high-pop' sample items with high popularity as priority
                        'low-pop' sample items with low popularity as prority
        sample_ratio : float, scope [0, 1], it determines the ratio that the other sample method except 'uniform' occupied, default is 0
        """
        super(BasicNegtiveSampler, self).__init__(args)
        self.user_num = args.num_users
        self.num_ng = args.num_ng
        self.inter_name = 'click'
        self.sample_method = args.sample_method
        self.sample_ratio = args.sample_ratio
        self.loss_type = args.loss_type.upper()

        assert self.sample_method in ['uniform', 'low-pop',
                                      'high-pop'], f'Invalid sampling method: {self.sample_method}'
        assert 0 <= self.sample_ratio <= 1, 'Invalid sample ratio value'

        self.df = df
        self.pop_prob = None

        if self.sample_method in ['high-pop', 'low-pop']:
            pop = df.groupby(self.iid_name).size()
            # rescale to [0, 1]
            pop /= pop.sum()
            pop = pop ** (3. / 4.)
            if self.sample_method == 'high-pop':
                norm_pop = np.zeros(self.item_num)
                norm_pop[pop.index] = pop.values
            if self.sample_method == 'low-pop':
                norm_pop = np.ones(self.item_num)
                norm_pop[pop.index] = (1 - pop.values)
            self.pop_prob = norm_pop / norm_pop.sum()

    def sampling(self):
        if self.num_ng == 0:
            self.df['neg_set'] = self.df.apply(lambda x: [], axis=1)
            if self.loss_type in ['CL', 'SL']:
                return self.df[[self.uid_name, self.iid_name, self.inter_name]].values.astype(np.int32)
            else:
                raise NotImplementedError('loss function (BPR, TL, HL) need num_ng > 0')

        js = np.zeros((self.user_num, self.num_ng), dtype=np.int32)
        if self.sample_method in ['low-pop', 'high-pop']:
            other_num = int(self.sample_ratio * self.num_ng)
            uniform_num = self.num_ng - other_num

            for u in tqdm(range(self.user_num)):
                past_inter = list(self.ur[u])

                uni_negs = []
                for _ in range(uniform_num):
                    item = np.random.choice(self.item_num)
                    while item in past_inter or item in uni_negs:
                        item = np.random.choice(self.item_num)
                    uni_negs.append(item)
                uni_negs = np.array(uni_negs)
                # uni_negs = np.random.choice(
                #     np.setdiff1d(np.arange(self.item_num), past_inter),
                #     size=uniform_num
                # )
                other_negs = []
                for _ in range(other_num):
                    item = np.random.choice(self.item_num, p=self.pop_prob)
                    while item in past_inter or item in uni_negs or item in other_negs:
                        item = np.random.choice(self.item_num, p=self.pop_prob)
                    other_negs.append(item)
                other_negs = np.array(other_negs)
                # other_negs = np.random.choice(
                #     np.arange(self.item_num),
                #     size=other_num,
                #     p=self.pop_prob
                # )
                js[u] = np.concatenate((uni_negs, other_negs), axis=None)

        else:
            # all negative samples are sampled by uniform distribution
            for u in tqdm(range(self.user_num)):
                past_inter = list(self.ur[u])
                neg = []
                for _ in range(self.num_ng):
                    item = np.random.choice(self.item_num)
                    while item in past_inter or item in neg:
                        item = np.random.choice(self.item_num)
                    neg.append(item)
                js[u] = np.array(neg)
                # js[u] = np.random.choice(
                #     np.setdiff1d(np.arange(self.item_num), past_inter),
                #     size=self.num_ng
                # )

        self.df['neg_set'] = self.df[self.uid_name].agg(lambda u: js[u])

        if self.loss_type.upper() in ['CL', 'SL']:
            point_pos = self.df[[self.uid_name, self.iid_name, self.inter_name]]
            point_neg = self.df[[self.uid_name, 'neg_set', self.inter_name]].copy()
            point_neg[self.inter_name] = 0
            point_neg = point_neg.explode('neg_set')
            return np.vstack([point_pos.values, point_neg.values]).astype(np.int32)
        elif self.loss_type.upper() in ['BPR', 'HL', 'TL']:
            self.df = self.df[[self.uid_name, self.iid_name, 'neg_set']].explode('neg_set')
            return self.df.values.astype(np.int32)
        else:
            raise NotImplementedError


class SkipGramNegativeSampler(AbstractSampler):
    def __init__(self, df, args, discard=False):
        '''
        skip-gram negative sampling class for <target_i, context_i, label>

        Parameters
        ----------
        df : pd.DataFrame
            training set
        rho : float, optional
            threshold to discard word in a sequence, by default 1e-5
        context_window: int, context range around target
        train_ur: dict, ground truth for each user in train set
        item_num: int, the number of items
        '''
        super(SkipGramNegativeSampler, self).__init__(args)
        self.context_window = args.context_window

        word_frequecy = df[self.iid_name].value_counts()
        prob_discard = 1 - np.sqrt(args.rho / word_frequecy)

        if discard:
            rnd_p = np.random.uniform(low=0., high=1., size=len(df))
            discard_p_per_item = df[self.iid_name].map(prob_discard).values
            df = df[rnd_p >= discard_p_per_item]

        self.train_seqs = self._build_seqs(df)

    def sampling(self):
        sgns_samples = []

        for u, seq in self.train_seqs.iteritems():
            past_inter = list(self.ur[u])
            cands = np.setdiff1d(np.arange(self.item_num), past_inter)

            for i in range(len(seq)):
                target = seq[i]
                # generate positive sample
                context_list = []
                j = i - self.context_window
                while j <= i + self.context_window and j < len(seq):
                    if j >= 0 and j != i:
                        context_list.append(seq[j])
                        sgns_samples.append([target, seq[j], 1])
                    j += 1
                # generate negative sample
                num_ng = len(context_list)
                for neg_item in np.random.choice(cands, size=num_ng):
                    sgns_samples.append([target, neg_item, 0])

        return np.array(sgns_samples)

    def _build_seqs(self, df):
        train_seqs = df.groupby(self.uid_name)[self.iid_name].agg(list)

        return train_seqs