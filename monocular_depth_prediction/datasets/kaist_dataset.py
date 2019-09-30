import os
import numpy as np
import PIL.Image as pil
from torch.utils.data import DataLoader

from datasets.mono_dataset import pil_loader, MonoDataset

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
        color = self.loader(self.get_image_path(frame_index, line))
        print('getting color')
        if do_flip:
            print('flipping)
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
            print('flipped!')
        print('after flip')
        return color

    def get_image_path(self, frame_index, line):
        f_str = "{:09d}{}".format(frame_index, self.img_ext)
        path = line.split('/')[:-1]
        path = '/'.join(path)
        image_path = os.path.join(path, "THER_{}".format(f_str))
        print(image_path)
        return image_path
    
    def check_depth(self):
        # check if depth file exists for an image

        return False
    
    def get_depth(self, folder, frame_index, side, do_flip):
        calib_path = os.path.join(self.data_path, folder.split("/")[0])

        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt