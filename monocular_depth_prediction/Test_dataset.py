from datasets.flir_dataset import FlirDataset
from datasets.kaist_dataset import KAIST_Dataset
import os
from torch.utils.data import DataLoader


if __name__ == "__main__":
    
    data_path = '/groups/mshah/data/KAIST_multispectral/'
    data_set = 'KAIST'
    img_ext = '.jpg
    
    datasets_dict = {'kitti': KITTIRAWDataset,
                     'kitti_odom': KITTIOdomDataset,
                     'FLIR': FlirDataset,
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
    train_loader = DataLoader(train_dataset, 12, True, num_workers=12, pin_memory=True, drop_last=True)
    
    val_dataset = dataset(data_path, val_filenames, 448, 512, [0, -1, 1], 4, is_train = False, img_ext=img_ext, thermal = thermal)
    val_loader = DataLoader(val_dataset, 12, True, num_workers=12, pin_memory=True, drop_last=True)
    val_iter = iter(val_loader)
    
    if dataset.startswith('kitti'):
            print("Using kitti")
        else:
            print("Using dataset:\n  ", dataset)
        
    print("There are {:d} training items and {:d} validation items\n".format(
        len(train_dataset), len(val_dataset)))
    
    for batch_idx, inputs in enumerate(train_loader):
            print(batch_idx)
    
