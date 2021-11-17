import ctypes

import torch

from .hdmlpvision import HDMLPVisionDataset
from hdmlp import hdmlp
import hdmlp.lib.transforms


class HDMLPCosmoFlow(HDMLPVisionDataset):
    """A dataloader for CosmoFlow binary files.

    Args:
        root (string): Root directory path.
        hdmlp_job (hdmlp.Job): Configured HDMLP job for the dataset.

    """

    def __init__(self, root, hdmlp_job: hdmlp.Job,
                 sample_shape=None, label_shape=(1,)):
        super(HDMLPCosmoFlow, self).__init__(root)

        self.sample_shape = tuple(sample_shape)
        self.label_shape = label_shape
        self.job = hdmlp_job
        self.hdmlp_transforms = hdmlp_job.get_transforms()
        self.job.setup()
        self.job_destroyed = False

    def __getitem__(self, item):
        num_items = 1
        if not isinstance(item, slice) and len(self.hdmlp_transforms) > 0:
            raise ValueError(
                "Must provide range in batch mode / with transformations")
        if isinstance(item, slice) and (len(self.hdmlp_transforms) > 0 or self.job.collate_data):
            num_items = item.stop - item.start
        if len(self.hdmlp_transforms) == 0 and not self.job.collate_data:
            label, sample = self.job.get(num_items, True, self.sample_shape,
                                         ctypes.c_short,
                                         is_string_label=False,
                                         label_shape=self.label_shape,
                                         fixed_label_len=16)
        elif self.job.collate_data:
            label, sample = self.job.get(
                num_items, True, (num_items,) + self.sample_shape,
                ctypes.c_short, False, self.label_shape, fixed_label_len=16)
            sample = torch.from_numpy(sample)
        else:
            w_out, h_out, c_out = self.job.get_transformed_dims()
            label, sample = self.job.get(
                num_items, True, (num_items, h_out, w_out, c_out),
                ctypes.c_short, False, self.label_shape, fixed_label_len=16)
            sample = torch.from_numpy(sample)
        return sample, label

    def __len__(self):
        return self.job.length()

    def __del__(self):
        if not self.job_destroyed:
            # Allows destructor to be called multiple times
            self.job_destroyed = True
            self.job.destroy()

    def get_job(self):
        return self.job
