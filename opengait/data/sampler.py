import math
import random
from collections import Counter

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data as tordata

try:
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.under_sampling import TomekLinks
except ImportError:
    RandomOverSampler = None
    RandomUnderSampler = None
    TomekLinks = None


class TripletSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, batch_size, batch_shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        if len(self.batch_size) != 2:
            raise ValueError(
                "batch_size should be (P x K) not {}".format(batch_size))
        self.batch_shuffle = batch_shuffle

        self.world_size = dist.get_world_size()
        if (self.batch_size[0]*self.batch_size[1]) % self.world_size != 0:
            raise ValueError("World size ({}) is not divisible by batch_size ({} x {})".format(
                self.world_size, batch_size[0], batch_size[1]))
        self.rank = dist.get_rank()

    def __iter__(self):
        while True:
            sample_indices = []
            pid_list = sync_random_sample_list(
                self.dataset.label_set, self.batch_size[0])

            for pid in pid_list:
                indices = self.dataset.indices_dict[pid]
                indices = sync_random_sample_list(
                    indices, k=self.batch_size[1])
                sample_indices += indices

            if self.batch_shuffle:
                sample_indices = sync_random_sample_list(
                    sample_indices, len(sample_indices))

            total_batch_size = self.batch_size[0] * self.batch_size[1]
            total_size = int(math.ceil(total_batch_size /
                                       self.world_size)) * self.world_size
            sample_indices += sample_indices[:(
                total_batch_size - len(sample_indices))]

            sample_indices = sample_indices[self.rank:total_size:self.world_size]
            yield sample_indices

    def __len__(self):
        return len(self.dataset)


def sync_random_sample_list(obj_list, k, common_choice=False):
    if common_choice:
        idx = random.choices(range(len(obj_list)), k=k) 
        idx = torch.tensor(idx)
    if len(obj_list) < k:
        idx = random.choices(range(len(obj_list)), k=k)
        idx = torch.tensor(idx)
    else:
        idx = torch.randperm(len(obj_list))[:k]
    if torch.cuda.is_available():
        idx = idx.cuda()
    torch.distributed.broadcast(idx, src=0)
    idx = idx.tolist()
    return [obj_list[i] for i in idx]


class InferenceSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

        self.size = len(dataset)
        indices = list(range(self.size))

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        if batch_size % world_size != 0:
            raise ValueError("World size ({}) is not divisible by batch_size ({})".format(
                world_size, batch_size))

        if batch_size != 1:
            complement_size = math.ceil(self.size / batch_size) * \
                batch_size
            indices += indices[:(complement_size - self.size)]
            self.size = complement_size

        batch_size_per_rank = int(self.batch_size / world_size)
        indx_batch_per_rank = []

        for i in range(int(self.size / batch_size_per_rank)):
            indx_batch_per_rank.append(
                indices[i*batch_size_per_rank:(i+1)*batch_size_per_rank])

        self.idx_batch_this_rank = indx_batch_per_rank[rank::world_size]

    def __iter__(self):
        yield from self.idx_batch_this_rank

    def __len__(self):
        return len(self.dataset)


class CommonSampler(tordata.sampler.Sampler):
    def __init__(self,dataset,batch_size,batch_shuffle):

        self.dataset = dataset
        self.size = len(dataset)
        self.batch_size = batch_size
        if isinstance(self.batch_size,int)==False:
            raise ValueError(
                "batch_size shoude be (B) not {}".format(batch_size))
        self.batch_shuffle = batch_shuffle
        
        self.world_size = dist.get_world_size()
        if self.batch_size % self.world_size !=0:
            raise ValueError("World size ({}) is not divisble by batch_size ({})".format(
                self.world_size, batch_size))
        self.rank = dist.get_rank() 
    
    def __iter__(self):
        while True:
            indices_list = list(range(self.size))
            sample_indices = sync_random_sample_list(
                    indices_list, self.batch_size, common_choice=True)
            total_batch_size =  self.batch_size
            total_size = int(math.ceil(total_batch_size /
                                       self.world_size)) * self.world_size
            sample_indices += sample_indices[:(
                total_batch_size - len(sample_indices))]
            sample_indices = sample_indices[self.rank:total_size:self.world_size]
            yield sample_indices

    def __len__(self):
        return len(self.dataset)


class ImbalancedSampler(tordata.sampler.Sampler):
    """Sequence-index sampler backed by imbalanced-learn.

    This sampler resamples sequence indices before collation, so it keeps the
    original gait sequence tensors intact and only changes how often each class
    appears during training.
    """

    def __init__(self, dataset, batch_size, batch_shuffle=False, balance_by='type',
                 sampling_method='RandomOverSampler', sampling_strategy='auto', random_state=0):
        self.dataset = dataset
        self.size = len(dataset)
        self.batch_size = batch_size
        if isinstance(self.batch_size, int) is False:
            raise ValueError(
                "batch_size shoude be (B) not {}".format(batch_size))
        self.batch_shuffle = batch_shuffle
        self.balance_by = balance_by
        self.sampling_method = sampling_method
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state

        self.world_size = dist.get_world_size()
        if self.batch_size % self.world_size != 0:
            raise ValueError("World size ({}) is not divisble by batch_size ({})".format(
                self.world_size, batch_size))
        self.rank = dist.get_rank()

        self._base_indices = self._build_resampled_index_pool()

    def _get_balance_labels(self):
        if self.balance_by in ['type', 'types', 'class']:
            return list(self.dataset.types_list)
        if self.balance_by in ['label', 'labels', 'identity']:
            return list(self.dataset.label_list)
        raise ValueError(
            f"balance_by should be one of type/types/class/label/labels/identity but got {self.balance_by}")

    def _get_sampler(self):
        if self.sampling_method == 'RandomOverSampler':
            if RandomOverSampler is None:
                raise ImportError("imbalanced-learn is required for ImbalancedSampler. Please install it with `pip install imbalanced-learn`.")
            return RandomOverSampler
        if self.sampling_method == 'RandomUnderSampler':
            if RandomUnderSampler is None:
                raise ImportError("imbalanced-learn is required for ImbalancedSampler. Please install it with `pip install imbalanced-learn`.")
            return RandomUnderSampler
        if self.sampling_method == 'TomekLinks':
            if TomekLinks is None:
                raise ImportError("imbalanced-learn is required for ImbalancedSampler. Please install it with `pip install imbalanced-learn`.")
            return TomekLinks
        raise ValueError(
            "sampling_method should be 'RandomOverSampler', 'RandomUnderSampler' or 'TomekLinks', got {}".format(self.sampling_method))

    def _build_resampled_index_pool(self):
        labels = self._get_balance_labels()
        if len(labels) == 0:
            raise ValueError("ImbalancedSampler received an empty dataset.")

        label_to_id = {lab: idx for idx, lab in enumerate(dict.fromkeys(labels))}
        encoded_labels = np.asarray([label_to_id[lab] for lab in labels], dtype=np.int64)

        # We keep an index-based pool here so the returned rows are always valid
        # original dataset indices. For TomekLinks this is an approximation because
        # the sampler does not have access to the real feature space at this stage.
        index_features = np.arange(len(labels), dtype=np.int64).reshape(-1, 1)

        sampler_kwargs = {
            'sampling_strategy': self.sampling_strategy,
        }
        # Not every imbalanced-learn sampler supports random_state.
        if self.sampling_method in ['RandomOverSampler', 'RandomUnderSampler']:
            sampler_kwargs['random_state'] = self.random_state

        sampler = self._get_sampler()(**sampler_kwargs)
        resampled_indices, _ = sampler.fit_resample(index_features, encoded_labels)
        resampled_indices = resampled_indices.reshape(-1).astype(np.int64).tolist()

        # Keep a small sanity check in the logs without changing the training flow.
        if self.rank == 0:
            original_dist = Counter(labels)
            resampled_dist = Counter(labels[idx] for idx in resampled_indices)
            print("[ImbalancedSampler] original distribution:", dict(original_dist))
            print("[ImbalancedSampler] resampled distribution:", dict(resampled_dist))

        return resampled_indices

    def __iter__(self):
        while True:
            epoch_indices = list(self._base_indices)
            if self.batch_shuffle:
                random.shuffle(epoch_indices)

            if not epoch_indices:
                raise ValueError("ImbalancedSampler has no indices to sample from.")

            total_batch_size = self.batch_size
            epoch_size = int(math.ceil(len(epoch_indices) / total_batch_size)) * total_batch_size
            epoch_indices += epoch_indices[:(epoch_size - len(epoch_indices))]

            total_size = int(math.ceil(total_batch_size / self.world_size)) * self.world_size
            for start in range(0, epoch_size, total_batch_size):
                batch_indices = epoch_indices[start:start + total_batch_size]
                batch_indices += batch_indices[:(total_batch_size - len(batch_indices))]
                batch_indices = batch_indices[self.rank:total_size:self.world_size]
                yield batch_indices

    def __len__(self):
        return int(math.ceil(len(self._base_indices) / self.batch_size))

# **************** For GaitSSB ****************
# Fan, et al: Learning Gait Representation from Massive Unlabelled Walking Videos: A Benchmark, T-PAMI2023
import random
class BilateralSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, batch_size, batch_shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.batch_shuffle = batch_shuffle

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        self.dataset_length = len(self.dataset)
        self.total_indices = list(range(self.dataset_length))

    def __iter__(self):
        random.shuffle(self.total_indices)
        count = 0
        batch_size = self.batch_size[0] * self.batch_size[1]
        while True:
            if (count + 1) * batch_size >= self.dataset_length:
                count = 0
                random.shuffle(self.total_indices)

            sampled_indices = self.total_indices[count*batch_size:(count+1)*batch_size]
            sampled_indices = sync_random_sample_list(sampled_indices, len(sampled_indices))

            total_size = int(math.ceil(batch_size / self.world_size)) * self.world_size
            sampled_indices += sampled_indices[:(batch_size - len(sampled_indices))]

            sampled_indices = sampled_indices[self.rank:total_size:self.world_size]
            count += 1

            yield sampled_indices * 2

    def __len__(self):
        return len(self.dataset)

class ClassBalancedSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, batch_size, batch_shuffle=False):
        self.dataset = dataset
        self.size = len(dataset)
        self.batch_size = batch_size
        if isinstance(self.batch_size, int) is False:
            raise ValueError(
                "batch_size shoude be (B) not {}".format(batch_size))
        self.batch_shuffle = batch_shuffle

        self.world_size = dist.get_world_size()
        if self.batch_size % self.world_size != 0:
            raise ValueError("World size ({}) is not divisble by batch_size ({})".format(
                self.world_size, self.batch_size))
        self.rank = dist.get_rank()

        # 按类别分组样本索引
        self.class_to_indices = {}
        for idx, cls in enumerate(self.dataset.types_list):
            self.class_to_indices.setdefault(cls, []).append(idx)
        self.class_list = list(self.class_to_indices.keys())

    def __iter__(self):
        while True:
            # 每个 batch 先均衡覆盖类别，再补齐
            indices = []
            for cls in self.class_list:
                pool = self.class_to_indices[cls]
                if not pool:
                    continue
                indices.append(random.choice(pool))
                if len(indices) >= self.batch_size:
                    break

            # 补齐到 batch_size
            while len(indices) < self.batch_size:
                cls = random.choice(self.class_list)
                pool = self.class_to_indices[cls]
                indices.append(random.choice(pool))

            if self.batch_shuffle:
                random.shuffle(indices)

            total_batch_size = self.batch_size
            total_size = int(math.ceil(total_batch_size /
                                       self.world_size)) * self.world_size
            indices += indices[:(total_batch_size - len(indices))]
            indices = indices[self.rank:total_size:self.world_size]
            yield indices

    def __len__(self):
        return len(self.dataset)
