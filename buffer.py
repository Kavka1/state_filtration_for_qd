from typing import Dict, List, Tuple
import numpy as np
from collections import deque
import random


class CoefficientBuffer(object):
    def __init__(self, coeff_buffer_size: int, num_coeff: int, coeff_range: List) -> None:
        self.num_coeff = num_coeff
        self.size = coeff_buffer_size
        self.coeff_range = coeff_range
        self.coeff_min, self.coeff_max = [item[0] for item in coeff_range], [item[1] for item in coeff_range]
        self.buffer = deque(maxlen=coeff_buffer_size)

    def _sample_coeff(self, sample_size: int) -> List[List]:
        assert len(self.buffer) > sample_size, "The coefficient buffer size is smaller than sample size"    
        samples = random.sample(self.buffer, sample_size)
        return samples
    
    def _mutate_sampled_coeff(self, sample_coeff: List[List], noise_num: int) -> List[List]:
        mutated_coeff = []
        np_sample_coeff = np.array(sample_coeff, dtype=np.float32)
        for i in range(noise_num):
            noise = [
                np.random.uniform(
                    -(self.coeff_range[j][1] - self.coeff_range[j][0]) / 2, (self.coeff_range[j][1] - self.coeff_range[j][0]) / 2
                ) 
                for j in range(len(self.coeff_range))
            ]
            offspring_coeff = np_sample_coeff + noise
            #offspring_coeff = np.array(
            #    [
            #        [np.random.uniform(self.coeff_range[j][0], self.coeff_range[j][1]) for j in range(len(self.coeff_range))]
            #        for _ in range(len(sample_coeff))
            #    ]
            ##)
            offspring_coeff = np.clip(offspring_coeff, self.coeff_min, self.coeff_max)
            mutated_coeff += offspring_coeff.tolist()
        return sample_coeff + mutated_coeff

    def draw_new_coeff(self, sample_size: int, noise_num: int) -> List[List]:
        sample_coeff = self._sample_coeff(sample_size)
        offspring_coeff = self._mutate_sampled_coeff(sample_coeff, noise_num)
        return offspring_coeff

    def store(self, new_coeff: List[List]) -> None:
        for coeff in new_coeff:
            self.buffer.append(coeff)

    def warmup(self) -> None:
        coeff = []
        for i in range(self.size):
            c = [np.random.random() * (self.coeff_max[i] - self.coeff_min[i]) + self.coeff_min[i] for i in range(self.num_coeff)]
            coeff.append(c)
        self.store(coeff)


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
        obs, a, r, done, obs_, eps = zip(*batch)
        obs, a, r, done, obs_, eps = np.stack(obs, 0), np.stack(a, 0), np.array(r), np.array(done), np.stack(obs_, 0), np.stack(eps, 0)
        return obs, a, r, done, obs_, eps

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