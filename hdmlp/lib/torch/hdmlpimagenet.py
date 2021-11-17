import os
import torch
from hdmlp import hdmlp
from .hdmlpfolder import HDMLPImageFolder

META_FILE = "meta.bin"


class HDMLPImageNet(HDMLPImageFolder):
    """`ImageNet <http://image-net.org/>`_ 2012 Classification Dataset.

    Args:
        root (string): Root directory of the ImageNet Dataset.
        split (string, optional): The dataset split, supports ``train``, or ``val``.
        hdmlp_job (hdmlp.Job): Configured HDMLP job for the dataset.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.

     Attributes:
        classes (list): List of the class name tuples.
        wnids (list): List of the WordNet IDs.
        wnid_to_idx (dict): Dict with items (wordnet_id, class_index).
        imgs (list): List of (image path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, hdmlp_job: hdmlp.Job, transform = None, target_transform = None, split='train', devkit_root = None):
        root = self.root = os.path.expanduser(root)
        self.split = split

        if devkit_root is None:
            devkit_root = root
        parse_devkit_archive(devkit_root)
        wnid_to_classes = load_meta_file(devkit_root)[0]

        super(HDMLPImageNet, self).__init__(self.split_folder, hdmlp_job, transform, target_transform)
        print("Init finished")
        self.root = root

        self.wnids = self.classes
        self.wnid_to_idx = self.class_to_idx
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]

    @property
    def split_folder(self):
        return os.path.join(self.root, self.split)

    def extra_repr(self):
        return "Split: {split}".format(**self.__dict__)


def load_meta_file(root, file=None):
    if file is None:
        file = META_FILE
    file = os.path.join(root, file)

    if os.path.exists(file):
        return torch.load(file)
    else:
        msg = ("The meta file {} is not present in the root directory"
               "This file is automatically created by the ImageNet dataset.")
        raise RuntimeError(msg.format(file, root))


def parse_devkit_archive(root):
    """Parse the devkit archive of the ImageNet2012 classification dataset and save
    the meta information in a binary file.

    Args:
        root (str): Root directory containing the devkit archive
    """
    import scipy.io as sio

    def parse_meta_mat(devkit_root):
        metafile = os.path.join(devkit_root, "data", "meta.mat")
        meta = sio.loadmat(metafile, squeeze_me=True)['synsets']
        nums_children = list(zip(*meta))[4]
        meta = [meta[idx] for idx, num_children in enumerate(nums_children)
                if num_children == 0]
        idcs, wnids, classes = list(zip(*meta))[:3]
        classes = [tuple(clss.split(', ')) for clss in classes]
        idx_to_wnid = {idx: wnid for idx, wnid in zip(idcs, wnids)}
        wnid_to_classes = {wnid: clss for wnid, clss in zip(wnids, classes)}
        return idx_to_wnid, wnid_to_classes

    def parse_val_groundtruth_txt(devkit_root):
        file = os.path.join(devkit_root, "data",
                            "ILSVRC2012_validation_ground_truth.txt")
        with open(file, 'r') as txtfh:
            val_idcs = txtfh.readlines()
        return [int(val_idx) for val_idx in val_idcs]



    devkit_root = os.path.join(root, "ILSVRC2012_devkit_t12")
    idx_to_wnid, wnid_to_classes = parse_meta_mat(devkit_root)
    val_idcs = parse_val_groundtruth_txt(devkit_root)
    val_wnids = [idx_to_wnid[idx] for idx in val_idcs]

    torch.save((wnid_to_classes, val_wnids), os.path.join(root, META_FILE))