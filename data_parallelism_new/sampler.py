import math

import torch
from torch.utils.data import DistributedSampler


class UnevenDistributedSampler(DistributedSampler):
    # customize the __iter__ method
    def __init__(self, dataset, num_replicas=None, rank=None, split_ratio_list=None):
        super().__init__(dataset=dataset, num_replicas=num_replicas, rank=rank)
        if split_ratio_list is None:
            split_ratio_list = [0.5, 0.5]
        if len(split_ratio_list) != num_replicas:
            raise ValueError("the num_replicas must equal the length of split_ratio_list")
        if sum(split_ratio_list) != 1:
            raise ValueError("the sum of split_ratio_list must equal 1")
        self.split_ratio_list = split_ratio_list

    def __iter__(self):
        """
        # indices is a list of element index of a dataset
        """
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        if self.rank == 0:  # data 0-59 on node 0
            indices = indices[0:40000]
        if self.rank == 1:  # data 60-99 on node 1
            indices = indices[40000:50000]

        return iter(indices)
