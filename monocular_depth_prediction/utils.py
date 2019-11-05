# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import os
import hashlib
import zipfile
from six.moves import urllib


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d


def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s


def sec_to_hm_str(t):
    """Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)


def download_model_if_doesnt_exist(model_name):
    """If pretrained kitti model doesn't exist, download and unzip it
    """
    # values are tuples of (<google cloud URL>, <md5 checksum>)
    download_paths = {
        "mono_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_640x192.zip",
             "a964b8356e08a02d009609d9e3928f7c"),
        "stereo_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_640x192.zip",
             "3dfb76bcff0786e4ec07ac00f658dd07"),
        "mono+stereo_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_640x192.zip",
             "c024d69012485ed05d7eaa9617a96b81"),
        "mono_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_no_pt_640x192.zip",
             "9c2f071e35027c895a4728358ffc913a"),
        "stereo_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_no_pt_640x192.zip",
             "41ec2de112905f85541ac33a854742d1"),
        "mono+stereo_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_no_pt_640x192.zip",
             "46c3b824f541d143a45c37df65fbab0a"),
        "mono_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_1024x320.zip",
             "0ab0766efdfeea89a0d9ea8ba90e1e63"),
        "stereo_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_1024x320.zip",
             "afc2f2126d70cf3fdf26b550898b501a"),
        "mono+stereo_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_1024x320.zip",
             "cdc5fc9b23513c07d5b19235d9ef08f7"),
        }

    if not os.path.exists("models"):
        os.makedirs("models")

    model_path = os.path.join("models", model_name)

    def check_file_matches_md5(checksum, fpath):
        if not os.path.exists(fpath):
            return False
        with open(fpath, 'rb') as f:
            current_md5checksum = hashlib.md5(f.read()).hexdigest()
        return current_md5checksum == checksum

    # see if we have the model already downloaded...
    if not os.path.exists(os.path.join(model_path, "encoder.pth")):

        model_url, required_md5checksum = download_paths[model_name]

        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("-> Downloading pretrained model to {}".format(model_path + ".zip"))
            urllib.request.urlretrieve(model_url, model_path + ".zip")

        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("   Failed to download a file which matches the checksum - quitting")
            quit()

        print("   Unzipping model...")
        with zipfile.ZipFile(model_path + ".zip", 'r') as f:
            f.extractall(model_path)

        print("   Model unzipped to {}".format(model_path))

def get_filenames(dataset, data_path, split):
    train_filenames = []
    val_filenames = []
    thermal = False
    if dataset == 'FLIR':
        train_files = os.listdir(os.path.join(data_path, 'train/PreviewData/'))
        train_files.sort()
        train_filenames.extend(os.path.join(data_path, 'train/PreviewData/') +
                               file for file in train_files[1:-1])

        video_files = os.listdir(os.path.join(data_path, 'video/PreviewData/'))
        video_files.sort()
        train_filenames.extend(os.path.join(data_path, 'video/PreviewData/') +
                               file for file in video_files[1:-1])

        val_files = os.listdir(os.path.join(data_path, 'valid/PreviewData/'))
        val_files.sort()
        val_filenames.extend(os.path.join(data_path, 'valid/PreviewData/') +
                             file for file in val_files[1:-1])
        thermal = True
    elif dataset == 'KAIST':
        train_files = os.path.join(data_path, 'training')

        campus_train = os.listdir(os.path.join(train_files, 'Campus/THERMAL/'))
        campus_train.sort()
        residential_train = os.listdir(os.path.join(train_files, 'Residential/THERMAL/'))
        residential_train.sort()
        urban_train = os.listdir(os.path.join(train_files, 'Urban/THERMAL/'))
        urban_train.sort()

        train_filenames.extend(os.path.join(train_files, 'Campus/THERMAL/') +
                               file for file in campus_train[1:-1])
        train_filenames.extend(os.path.join(train_files, 'Residential/THERMAL/') +
                               file for file in residential_train[1:-1])
        train_filenames.extend(os.path.join(train_files, 'Urban/THERMAL/') +
                               file for file in urban_train[1:-1])

        val_files = os.path.join(data_path, 'testing')

        campus_val = os.listdir(os.path.join(val_files, 'Campus/THERMAL/'))
        campus_val.sort()
        residential_val = os.listdir(os.path.join(val_files, 'Residential/THERMAL/'))
        residential_val.sort()
        urban_val = os.listdir(os.path.join(val_files, 'Urban/THERMAL/'))
        urban_val.sort()

        val_filenames.extend(os.path.join(val_files, 'Campus/THERMAL/') +
                             file for file in campus_val[1:-1])
        val_filenames.extend(os.path.join(val_files, 'Residential/THERMAL/') +
                             file for file in residential_val[1:-1])
        val_filenames.extend(os.path.join(val_files, 'Urban/THERMAL/') +
                             file for file in urban_val[1:-1])
        thermal = True
    elif dataset == 'CREOL':
        train_path = os.path.join(data_path, 'training')
        train_sequences = os.listdir(train_path)

        for sequence in train_sequences:
            sequence_files = os.listdir(os.path.join(train_path, sequence))
            sequence_files.sort()
            train_filenames.extend(os.path.join(train_path, sequence, file) for file in sequence_files[1:-1])

        val_path = os.path.join(data_path, 'testing')
        val_sequences = os.listdir(val_path)

        for sequence in val_sequences:
            sequence_files  = os.listdir(os.path.join(val_path, sequence))
            sequence_files.sort()
            val_filenames.extend(os.path.join(val_path, sequence, file) for file in sequence_files[1:-1])

        thermal = True

    else:
        fpath = os.path.join(os.path.dirname(__file__), "splits", split, "{}_files.txt")
        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))

    return train_filenames, val_filenames, thermal

