import os
import numpy as np
import PIL.Image as pil
from torch.utils.data import DataLoader

from datasets.mono_dataset import *

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

    def check_depth(self):
        # FLIR dataset has no ground truth depth
        
        return False
    
    def get_color(self, frame_index, do_flip, line):
        color = self.loader(self.get_image_path(frame_index, line))
        
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
            
        return color

    def get_image_path(self, frame_index, line):
        f_str = "{:05d}{}".format(frame_index, self.img_ext)
        path = line.split('/')[:-1]
        path = '/'.join(path)
        
        if 'video' not in path:
            image_path = os.path.join(path, "FLIR_{}".format(f_str))
        else:
            image_path = os.path.join(path, "FLIR_video_{}".format(f_str))
        print(image_path)
        return image_path