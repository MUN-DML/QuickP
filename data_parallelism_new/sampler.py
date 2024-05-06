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
        indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        start = sum(self.split_ratio_list[:self.rank]) * len(self.dataset)  # type: ignore[arg-type]
        ratio = self.split_ratio_list[self.rank]
        length = len(self.dataset) * ratio  # type: ignore[arg-type]
        indices = indices[start: start+length]
        assert len(indices) == length
        return iter(indices)
