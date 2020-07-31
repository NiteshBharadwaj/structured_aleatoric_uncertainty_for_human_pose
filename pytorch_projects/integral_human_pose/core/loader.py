import numpy as np

import torch.utils.data as data

from common.utility.image_processing_cv import get_single_patch_sample

from common_pytorch.dataset.hm36 import from_mpii_to_hm36


class single_patch_Dataset(data.Dataset):
    def __init__(self, db, is_train, patch_width, patch_height, rect_3d_width, rect_3d_height,
                 batch_size, mean, std, aug_config, label_func, label_config):

        self.db = db[0].gt_db()

        self.num_samples = len(self.db)

        self.joint_num = db[0].joint_num

        self.is_train = is_train
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.rect_3d_width = rect_3d_width
        self.rect_3d_height = rect_3d_height
        self.mean = mean
        self.std = std
        self.aug_config = aug_config
        self.label_func = label_func
        self.label_config = label_config

        if self.is_train:
            self.do_augment = True
        else:
            self.do_augment = False
            # padding samples to match input_batch_size
            extra_db = len(self.db) % batch_size
            for i in range(0, batch_size - extra_db):
                self.db.append(self.db[i])

        self.db_length = len(self.db)

    def __getitem__(self, index):
        the_db = self.db[index]
        img_patch, label, label_weight = \
            get_single_patch_sample(the_db['image'], the_db['center_x'], the_db['center_y'],
                                    the_db['width'], the_db['height'],
                                    the_db['joints_3d'].copy(), the_db['joints_3d_vis'].copy(),
                                    the_db['flip_pairs'].copy(), the_db['parent_ids'].copy(),
                                    self.patch_width, self.patch_height,
                                    self.rect_3d_width, self.rect_3d_height, self.mean, self.std,
                                    self.do_augment, self.aug_config,
                                    self.label_func, self.label_config)

        return img_patch.astype(np.float32), label.astype(np.float32), label_weight.astype(np.float32)

    def __len__(self):
        return self.db_length


class hm36_Dataset(single_patch_Dataset):
    def __init__(self, db, is_train, patch_width, patch_height, rect_3d_width, rect_3d_height, batch_size,
                 mean, std, aug_config, label_func, label_config):
        single_patch_Dataset.__init__(self, db, is_train, patch_width, patch_height, rect_3d_width,
                                      rect_3d_height, batch_size, mean, std, aug_config, label_func, label_config)

class mpii(single_patch_Dataset):
    def __init__(self, db, is_train, det_bbox_src, patch_width, patch_height, rect_3d_width, rect_3d_height, batch_size,
                 mean, std, aug_config, label_func, label_config):

        assert det_bbox_src == ''
        self.dbo = db
        self.db = db.gt_db()  # mpii

        self.num_samples = len(self.db)

        from_mpii_to_hm36(self.db)

        self.joint_num = db.joint_num

        self.is_train = is_train
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.rect_3d_width = rect_3d_width
        self.rect_3d_height = rect_3d_height
        self.mean = mean
        self.std = std
        self.aug_config = aug_config
        self.label_func = label_func
        self.label_config = label_config

        if self.is_train:
            self.do_augment = True
        else:
            self.do_augment = False
        self.to_train_indices = np.arange(self.num_samples)
        self.update_db()

    def update_db(self):
        self.num_samples = self.to_train_indices.shape[0]
        self.db_length = self.num_samples
        self.count = 0

    def update_indices(self, to_train_indices):
        self.to_train_indices = to_train_indices.copy()
        self.update_db()

    def __getitem__(self, index):
        idx = self.to_train_indices[index]
        the_db = self.db[idx]
        imagepath = the_db['image']
        img_patch, label, label_weight = \
            get_single_patch_sample(imagepath, the_db['center_x'], the_db['center_y'],
                                    the_db['width'], the_db['height'],
                                    the_db['joints_3d'].copy(), the_db['joints_3d_vis'].copy(),
                                    self.db[0]['flip_pairs'].copy(), self.db[0]['parent_ids'].copy(),
                                    self.patch_width, self.patch_height,
                                    self.rect_3d_width, self.rect_3d_height, self.mean, self.std,
                                    self.do_augment, self.aug_config,
                                    self.label_func, self.label_config)


        return img_patch.astype(np.float32), label.astype(np.float32), label_weight.astype(np.float32)

    def __len__(self):
        return self.db_length
class mpii_hm36_Dataset(data.Dataset):
    def __init__(self, db, is_train, det_bbox_src, patch_width, patch_height, rect_3d_width, rect_3d_height, batch_size,
                 mean, std, aug_config, label_func, label_config):

        assert det_bbox_src == ''

        self.db0 = db[0].gt_db()  # mpii
        self.db1 = db[1].gt_db()  # hm36

        self.num_samples0 = len(self.db0)
        self.num_samples1 = len(self.db1)

        from_mpii_to_hm36(self.db0)

        self.joint_num = db[1].joint_num

        self.is_train = is_train
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.rect_3d_width = rect_3d_width
        self.rect_3d_height = rect_3d_height
        self.mean = mean
        self.std = std
        self.aug_config = aug_config
        self.label_func = label_func
        self.label_config = label_config

        if self.is_train:
            self.do_augment = True
        else:
            assert 0, "testing not supported for mpii_hm36_Dataset"

        self.db_length = self.num_samples0 * 2

        self.count = 0
        self.idx = np.arange(self.num_samples1)
        np.random.shuffle(self.idx)

    def __getitem__(self, index):
        if index < self.num_samples0:
            the_db = self.db0[index]
        else:
            the_db = self.db1[self.idx[index - self.num_samples0]]

        img_patch, label, label_weight = \
            get_single_patch_sample(the_db['image'], the_db['center_x'], the_db['center_y'],
                                    the_db['width'], the_db['height'],
                                    the_db['joints_3d'].copy(), the_db['joints_3d_vis'].copy(),
                                    self.db1[0]['flip_pairs'].copy(), self.db1[0]['parent_ids'].copy(),
                                    self.patch_width, self.patch_height,
                                    self.rect_3d_width, self.rect_3d_height, self.mean, self.std,
                                    self.do_augment, self.aug_config,
                                    self.label_func, self.label_config)

        self.count = self.count + 1
        if self.count >= self.db_length:
            self.count = 0
            self.idx = np.arange(self.num_samples1)
            np.random.shuffle(self.idx)

        return img_patch.astype(np.float32), label.astype(np.float32), label_weight.astype(np.float32)

    def __len__(self):
        return self.db_length
