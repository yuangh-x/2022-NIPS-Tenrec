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
                while item in seen or item in samples:
                    item = np.random.choice(self.item_count + 1) #
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
            for item in popular_items_freq.keys():
                # if len(samples) == self.sample_size:
                #     break
                if item in seen:
                    if item not in temp:
                        continue
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