from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import itertools
from collections import Counter
import random


class Item2Vec(Dataset):
    def __init__(self, path, context_window, neg_samples, discard_threshold=1e-5, discard=False) -> None:
        super().__init__()
        self.path = path
        self.context_window = context_window
        self.discard_threshold = discard_threshold
        self.discard = discard
        self.neg_samples = neg_samples
        self.corpus = self._setup(self.path)
        self.training_pairs = self._sgns_sample_generator(self.corpus, self.context_window, self.neg_samples)

    def _setup(self, path):
        df = pd.read_csv(path, sep='\t', header=None, names=['user', 'item', 'rating', 'timestamp'], engine='python')
        vocab_size = df['item'].nunique()
        df['user'] -= 1
        df['item'] -= 1
        df = df.groupby('user')['item'].agg(list)
        return df.values

    def _choose_with_prob(self, prob):
        p = np.random.uniform(low=0.0, high=1.0)
        return False if p < prob else False

    def _sgns_sample_generator(self, corpus, context_window, neg_samples):
        vocab = itertools.chain.from_iterable(list(corpus))

        item_frequency = dict(Counter(vocab))
        item_discard_prob = {key : 1 - np.sqrt(self.discard_threshold/value) for (key, value) in item_frequency.items()}

        training_pairs = []

        for order in corpus:
            if self.discard:
                order = [item for item in order if self._choose_with_prob(item_discard_prob[item])]

            for i in range(len(order)):
                target_item = order[i]
                context_list = []

                #sampling positive pair
                j = i - self.context_window
                while j <= i + self.context_window and j < len(order):
                    if j >= 0 and j != i:
                        context_list.append(order[j])
                        training_pairs.append((target_item, order[j], 1))

                
                #sampling negative pair
                for _ in range(self.neg_samples):
                    neg_item = random.choice(vocab)
                    while neg_item in context_list:
                        neg_item = random.choice(vocab)
                    context_list.append(neg_item)
                    training_pairs.append((target_item, neg_item, 0))


    def __len__(self):
        return len(self.training_pair)

    def __getitem__(self, index):
        return self.training_pairs[index]

if __name__ == "__main__":
    dataset = Item2Vec('/home/anhnguyen68/WorkSpace/RecSys/data/u.data', context_window=2, neg_samples=4)

    print(len(dataset))
    print(dataset[237])