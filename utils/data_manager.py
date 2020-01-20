from __future__ import print_function, absolute_import
import os.path as osp

"""
This file is used for preparing the classification data.
Read the train and val data for training and validation.
"""

__all__ = ['init_img_dataset', 'Aircraft_Carrier']


class BaseFineGrainDataset(object):
    """
    Base class of Fine Grained Image Classification dataset
    """

    def get_imagedata_info(self, data):
        classes = []
        for _, label in data:
            classes.append(label)
        class_set = set(classes)
        num_classes = len(class_set)
        num_imgs = len(data)
        return num_classes, num_imgs

    def print_dataset_statistics(self, train, test):
        num_train_pids, num_train_imgs = self.get_imagedata_info(train)
        num_test_pids, num_test_imgs = self.get_imagedata_info(test)

        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images   ")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}      ".format(num_train_pids, num_train_imgs))
        print("  test     | {:5d} | {:8d}      ".format(num_test_pids, num_test_imgs))
        print("  ------------------------------")


class Aircraft_Carrier(BaseFineGrainDataset):
    def __init__(self, root=r'/home/deep/kk/data/FineGrained/Aircraft_Carrier/', verbose=True, **kwargs):
        super(Aircraft_Carrier, self).__init__()
        train, test = self._process_dir(root)
        if verbose:
            print("=> Aircraft_Carrier loaded")
            self.print_dataset_statistics(train, test)

        self.train = train
        self.test = test

        self.num_train_pids, self.num_train_imgs = self.get_imagedata_info(self.train)
        self.num_test_pids, self.num_test_imgs = self.get_imagedata_info(self.test)
        assert self.num_train_pids == self.num_test_pids
        self.num_cls = self.num_train_pids

    def _process_dir(self, root):
        images_train = osp.join(root, 'train_label.txt')
        images_test = osp.join(root, 'val_label.txt')

        train_dataset = []
        with open(images_train, 'r') as f:
            lines_images = f.readlines()
            for line in lines_images:
                strs = line.split(' ')
                image_path = strs[0]
                label = strs[1].strip()
                # print(label)
                image_info = [image_path, int(label)]
                train_dataset.append(image_info)

        test_dataset = []
        with open(images_test, 'r') as f:
            lines_train_test = f.readlines()
            for line in lines_train_test:
                strs = line.split(' ')
                image_path = strs[0]
                label = strs[1].strip()
                image_info = [image_path, int(label)]
                test_dataset.append(image_info)

        return train_dataset, test_dataset


"""Create dataset"""
__img_factory = {
    'air': Aircraft_Carrier,
}


def get_names():
    return list(__img_factory.keys())


def init_img_dataset(name, **kwargs):
    if name not in __img_factory.keys():
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, __img_factory.keys()))
    return __img_factory[name](**kwargs)
