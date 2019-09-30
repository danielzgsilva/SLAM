from __future__ import absolute_import, division, print_function
import os
from torch.utils.data import DataLoader
import PIL.Image as pil

import os
import random
import numpy as np
import copy
import re
import sys

import torch
import torch.utils.data as data
from torchvision import transforms


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with pil.open(f) as img:
            return img.convert('RGB')


class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
        thermal - differentiates between kitti and thermal datasets, needed for file reading
    """

    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='.jpg',
                 thermal=False):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = pil.ANTIALIAS

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext

        self.thermal = thermal

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        self.load_depth = self.check_depth()

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5 and not self.thermal
        do_flip = self.is_train and random.random() > 0.5

        if not self.thermal:
            # load kitti data from split files
            line = self.filenames[index].split()
            folder = line[0]

            if len(line) == 3:
                frame_index = int(line[1])
            else:
                frame_index = 0

            if len(line) == 3:
                side = line[2]
            else:
                side = None

            for i in self.frame_idxs:
                if i == "s":
                    other_side = {"r": "l", "l": "r"}[side]
                    inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip)
                else:
                    inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)
        else:
            # load images from thermal datasets
            line = self.filenames[index]
            frame_index = int(re.findall(r'\d+', line)[-1])

            for i in self.frame_idxs:
                try:
                    inputs[("color", i, -1)], path = self.get_color(frame_index + i, do_flip, line)
                    inputs[("path", i, -1)] = path
                except Exception as e:
                    print(e)

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
    #        del inputs[("color", i, -1)]
     #       del inputs[("color_aug", i, -1)]

        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        return inputs


class FlirDataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(FlirDataset, self).__init__(*args, **kwargs)

        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
        self.img_ext = '.jpeg'
        self.loader = pil_loader

    def check_depth(self):
        # FLIR dataset has no ground truth depth

        return False

    def get_color(self, frame_index, do_flip, line):
        path = self.get_image_path(frame_index, line)
        color = self.loader(path)

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color, path

    def get_image_path(self, frame_index, line):
        f_str = "{:05d}{}".format(frame_index, self.img_ext)
        path = line.split('/')[:-1]
        path = '/'.join(path)

        if 'video' not in path:
            image_path = os.path.join(path, "FLIR_{}".format(f_str))
        else:
            image_path = os.path.join(path, "FLIR_video_{}".format(f_str))
        sys.stderr.write(image_path)
        return image_path

class KAIST_Dataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(KAIST_Dataset, self).__init__(*args, **kwargs)

        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
        self.img_ext = '.jpg'

    def get_color(self, frame_index, do_flip, line):
        path = self.get_image_path(frame_index, line)
        color = self.loader(path)
        if do_flip:
            print('need flip')
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
            print('flipped!!')
        return color, path

    def get_image_path(self, frame_index, line):
        f_str = "{:09d}{}".format(frame_index, self.img_ext)
        path = line.split('/')[:-1]
        path = '/'.join(path)
        image_path = os.path.join(path, "THER_{}".format(f_str))
        sys.stderr.write(image_path)
        return image_path

    def check_depth(self):
        # check if depth file exists for an image

        return False


if __name__ == "__main__":
    
    data_path = '/groups/mshah/data/KAIST_multispectral/'
    data_set = 'KAIST'
    img_ext = '.jpg'
    
    datasets_dict = {'FLIR': FlirDataset,
                     'KAIST': KAIST_Dataset}
        
    dataset = datasets_dict[data_set]

    thermal = False
    if data_set == 'KAIST':
        train_files = os.path.join(data_path, 'training')
        train_filenames = []

        train_filenames.extend(os.path.join(train_files, 'Campus/THERMAL/') + 
                               file for file in os.listdir(os.path.join(train_files, 'Campus/THERMAL/'))[1:-1])
        train_filenames.extend(os.path.join(train_files, 'Residential/THERMAL/') + 
                               file for file in os.listdir(os.path.join(train_files, 'Residential/THERMAL/'))[1:-1])
        train_filenames.extend(os.path.join(train_files, 'Urban/THERMAL/') + 
                               file for file in os.listdir(os.path.join(train_files, 'Urban/THERMAL/'))[1:-1])

        val_files = os.path.join(data_path, 'testing')
        val_filenames = []

        val_filenames.extend(os.path.join(val_files, 'Campus/THERMAL/') + 
                               file for file in os.listdir(os.path.join(val_files, 'Campus/THERMAL/'))[1:-1])
        val_filenames.extend(os.path.join(val_files, 'Residential/THERMAL/') + 
                               file for file in os.listdir(os.path.join(val_files, 'Residential/THERMAL/'))[1:-1])
        val_filenames.extend(os.path.join(val_files, 'Urban/THERMAL/') + 
                               file for file in os.listdir(os.path.join(val_files, 'Urban/THERMAL/'))[1:-1])    
        thermal = True
    else:
        print('wrong dataset')

    num_train_samples = len(train_filenames)
    num_total_steps = num_train_samples // 12 * 20

    train_dataset = dataset(data_path, train_filenames, 448, 512, [0, -1, 1], 4, is_train = True, img_ext=img_ext, thermal = thermal)
    train_loader = DataLoader(train_dataset, 12, False, num_workers=12, pin_memory=True, drop_last=True)
    
    val_dataset = dataset(data_path, val_filenames, 448, 512, [0, -1, 1], 4, is_train = False, img_ext=img_ext, thermal = thermal)
    val_loader = DataLoader(val_dataset, 12, False, num_workers=12, pin_memory=True, drop_last=True)
    val_iter = iter(val_loader)
    
    if data_set.startswith('kitti'):
        print("Using kitti")
    else:
        print("Using dataset:\n  ", data_set)
        
    print("There are {:d} training items and {:d} validation items\n".format(len(train_dataset), len(val_dataset)))
    
    for batch_idx, inputs in enumerate(train_loader):
            print(inputs.keys())
            print(inputs[("path", -1, -1)])
            print(inputs[("path", -0, -1)])
            print(inputs[("path", 1, -1)])
    
