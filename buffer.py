from typing import Dict, List, Tuple
import numpy as np
from collections import deque
import random


class Buffer(object):
    def __init__(self, buffer_size: int) -> None:
        super(Buffer, self).__init__()
        self.size = buffer_size
        self.data = deque(maxlen=self.size)
    
    def store(self, trans: Tuple) -> None:
        self.data.append(trans)
    
    def save_batch_trans(self, trans_batch: List[Tuple]) -> None:
        for trans in trans_batch:
            self.data.append(trans)

    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.data, batch_size)
        all_items = zip(*batch)
        for item in all_items:
            item = np.stack(item, 0)
        return all_items

    def clear(self) -> None:
        self.data.clear()

    def __len__(self):
        return len(self.data)


class Dataset(object):
    # For PPO training
    def __init__(self, data: Dict) -> None:
        super(Dataset, self).__init__()
        self.data = data
        self.n = len(data['ret'])
        self._next_id = 0
        self.shuffle()

    def shuffle(self) -> None:
        perm = np.arange(self.n)
        np.random.shuffle(perm)
        for key in list(self.data.keys()):
            self.data[key] = self.data[key][perm]

    def next_batch(self, batch_size: int) -> Tuple:
        cur_id = self._next_id
        cur_batch_size = min(batch_size, self.n - cur_id)
        self._next_id += cur_batch_size

        batch = dict()
        for key in list(self.data.keys()):
            batch[key] = self.data[key][cur_id: cur_id + cur_batch_size]
        return batch

    def iterate_once(self, batch_size: int) -> Tuple:
        self.shuffle()
        while self._next_id < self.n:
            yield self.next_batch(batch_size)
        self._next_id = 0