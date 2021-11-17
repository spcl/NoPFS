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
import pickle

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)

class HDMLPDatasetFolder(HDMLPVisionDataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

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

    def __init__(self, root, hdmlp_job: hdmlp.Job, transform=None,
                 target_transform=None, filelist=None):
        super(HDMLPDatasetFolder, self).__init__(root, transform=transform,
                                                 target_transform=target_transform)
        if filelist is not None and os.path.exists(filelist):
            with open(filelist, 'rb') as fp:
                classes, class_to_idx, _ = pickle.load(fp)
        else:
            classes, class_to_idx = self._find_classes(self.root)

        self.job = hdmlp_job
        self.job.setup()
        self.job_destroyed = False

        self.classes = classes
        self.class_to_idx = class_to_idx

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        folder_label, sample = self.job.get()
        target = self.class_to_idx[folder_label]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return self.job.length()

    def __del__(self):
        if not self.job_destroyed:
            self.job_destroyed = True # Allows destructor to be called multiple times
            self.job.destroy()

    def get_job(self):
        return self.job

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_decode(object):
    img = Image.open(io.BytesIO(object))
    return img.convert('RGB')


class HDMLPImageFolder(HDMLPDatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        hdmlp_job (hdmlp.Job): Configured HDMLP job for the dataset.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
    """

    def __init__(self, root, hdmlp_job: hdmlp.Job, transform=None, target_transform=None, filelist=None):
        hdmlp_transforms = hdmlp_job.get_transforms()
        self.img_decode = not any(isinstance(hdmlp_transform, hdmlp.lib.transforms.ImgDecode) for hdmlp_transform in hdmlp_transforms)
        # For now, only support this being this last transform.
        self.chw = len(hdmlp_transforms) > 0 and isinstance(hdmlp_transforms[-1], hdmlp.lib.transforms.HWCtoCHW)
        super(HDMLPImageFolder, self).__init__(root, hdmlp_job,
                                               transform=transform,
                                               target_transform=target_transform,
                                               filelist=filelist)


    def __getitem__(self, item):
        num_items = 1
        if not isinstance(item, slice) and not self.img_decode:
            raise ValueError("Must provide range in batch mode / with transformations")
        if isinstance(item, slice) and not self.img_decode:
            num_items = item.stop - item.start
        if self.img_decode:
            folder_label, sample = self.job.get(num_items, False)
            sample = pil_decode(sample)
        else:
            w_out, h_out, c_out = self.job.get_transformed_dims()
            if self.chw:
                folder_label, sample = self.job.get(
                    num_items, True, (num_items, c_out, h_out, w_out),
                    ctypes.c_ubyte)
                sample = torch.from_numpy(sample)
            else:
                folder_label, sample = self.job.get(
                    num_items, True, (num_items, h_out, w_out, c_out))
                sample = torch.from_numpy(sample)
                sample = sample.permute(0, 3, 1, 2)
        if isinstance(folder_label, list):
            target = [self.class_to_idx[fl] for fl in folder_label]
        else:
            target = self.class_to_idx[folder_label]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
