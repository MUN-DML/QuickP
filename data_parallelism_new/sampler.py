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

        start = sum(self.split_ratio_list[:self.rank]) * len(indices)
        ratio = self.split_ratio_list[self.rank]
        length = len(indices) * ratio
        indices = indices[int(start): int(start + length)]
        assert len(indices) == length
        return iter(indices)


def get_uneven_loader(dataset, world_size, rank, num_workers, batch_size_list):
    assert len(batch_size_list) == world_size >= 2 > rank
    # create split_ratio_list based on the batch_size_list
    sum_of_world_size = sum(batch_size_list)
    split_ratio_list = [x / sum_of_world_size for x in batch_size_list]
    train_sampler = UnevenDistributedSampler(dataset=dataset, num_replicas=world_size, rank=rank,
                                             split_ratio_list=split_ratio_list)
    '''batch size will be unequal among all ranks. Thus, 
    it will be the format of [batch_size_1, batch_size_2, ...][rank_1, rank_2, ...]'''
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size_list[rank], shuffle=(train_sampler is None),
        num_workers=num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    return train_loader, train_sampler
