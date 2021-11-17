import ctypes

import torch

from .hdmlpvision import HDMLPVisionDataset
from hdmlp import hdmlp
import hdmlp.lib.transforms

from PIL import Image
from torchvision import transforms

import os
import os.path
import io


class HDMLPHDF5(HDMLPVisionDataset):
    """A dataloader to directly load / unpack HDF5 files

    Args:
        root (string): Root directory path.
        hdmlp_job (hdmlp.Job): Configured HDMLP job for the dataset.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        folder_to_idx (dict): Dict with items (folder_name, class_index)
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, hdmlp_job: hdmlp.Job,
                 sample_shape=None, label_shape=(1,)):
        super(HDMLPHDF5, self).__init__(root)

        self.sample_shape = sample_shape
        self.label_shape = label_shape
        self.job = hdmlp_job
        self.hdmlp_transforms = hdmlp_job.get_transforms()
        self.job.setup()
        self.job_destroyed = False


    def __getitem__(self, item):
        num_items = 1
        if not isinstance(item, slice) and len(self.hdmlp_transforms) > 0:
            raise ValueError("Must provide range in batch mode / with transformations")
        if isinstance(item, slice) and len(self.hdmlp_transforms) > 0:
            num_items = item.stop - item.start
        if len(self.hdmlp_transforms) == 0:
            label, sample = self.job.get(num_items, True, self.sample_shape,
                                         is_string_label=False,
                                         label_shape=self.label_shape)
        else:
            w_out, h_out, c_out = self.job.get_transformed_dims()
            label, sample = self.job.get(num_items, True, (num_items, h_out, w_out, c_out), False, self.label_shape)
            sample = torch.from_numpy(sample)
        return sample, label

    def __len__(self):
        return self.job.length()

    def __del__(self):
        if not self.job_destroyed:
            self.job_destroyed = True # Allows destructor to be called multiple times
            self.job.destroy()

    def get_job(self):
        return self.job
