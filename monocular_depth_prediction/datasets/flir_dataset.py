import os
import random
import numpy as np
import copy

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import PIL.Image as pil

from mono_dataset import pil_loader

class FlirDataset(data.Dataset):
    """
    MonoDataset Class is from https://github.com/nianticlabs/monodepth2
        Step 1. Randomly choose to either do a color augmentation or a flip augmentation on the set of images
        Step 2. They apply random alterations to the brightness, contrast, saturation and hue
        Step 3. Creates a dictionary of the resized version of the images based on the scales passed and Image.ANTIALIAS interpolation
        Step 4. Gets image next to it both before and after at native resolution and color.
            - How do we know which is left or right? If not stereo it doesn't matter, does frame index plus one of [0, -1, 1]
        Step 5: Create different scales by using 'K' pre-defined as well as Moore-Penrose psuedo-inverse
                K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        Returns a dictionary with the key indicating the type so there is either:
            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.
            <frame_id>: an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
            <scale> is an integer representing the scale of the image relative to the fullsize image:
                -1      images at native resolution as loaded from disk
                0       images resized to (self.width,      self.height     )
                1       images resized to (self.width // 2, self.height // 2)
                2       images resized to (self.width // 4, self.height // 4)
                3       images resized to (self.width // 8, self.height // 8)
        Example use:
        datasets.FLIRDataset(dir, filenames, encoder_dict['height'],
                            encoder_dict['width'],  [0], 4, is_train=False)
    Args:
        data_path (str): Root of the data the filenames are under
        filenames (list): List of filenames to read in as images
        height (int): Default is 192
        width (int): Default is 640
        frame_idxs (list): Default is [0, -1, 1]
        num_scales (int): Default is 4 meaning [0, 1, 2, 3] scales
        is_train (bool): Whether this is training or not
        img_ext (str): Image file extension type.
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height=192,
                 width=640,
                 frame_idxs=[0, -1, 1],
                 num_scales=4,
                 is_train=False,
                 img_ext='.jpeg'):
        super(FlirDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames  # ignore last and first one since can't get image before, just for testing for now cause don't fully understand
        self.filenames.sort()
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1242, 375)

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

        #do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[index]  # is reading from a text list
        frame_index = int(re.findall(r'\d+', line)[0])

        for i in self.frame_idxs:
            try:
                inputs[("color", i, -1)] = self.get_color(frame_index + i, do_flip)
            except:
                continue

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)
        
        for i in self.frame_idxs:  # gets rid of original raw values
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]
        return inputs

    def get_color(self, frame_index, do_flip):
        color = self.loader(self.get_image_path(frame_index))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_image_path(self, frame_index):
        f_str = "{:05d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(self.data_path, "FLIR_{}".format(f_str))
        print(image_path)
        return image_path