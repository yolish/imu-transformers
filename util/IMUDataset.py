from skimage.io import imread
from torch.utils.data import Dataset
import pandas as pd
from os.path import join
import numpy as np


class IMUDataset(Dataset):
    """
        A class representing a dataset IMU learning tasks
    """

    def __init__(self, dataset_path, labels_file):
        """
        :param dataset_path: (str) the path to the dataset
        :param labels_file: (str) a file with images and their path labels
        :return: an instance of the class
        """
        super(CameraPoseDataset, self).__init__()
        self.imu_signal, self.labels = read_labels_file(labels_file, dataset_path)

    def __len__(self):
        return self.poses.shape[0]

    def __getitem__(self, idx):
        sample = {'imu': self.imu_signal[idx],
                  'label': self.labels[idx]}
        return sample


def read_labels_file(labels_file, dataset_path):
    df = pd.read_csv(labels_file)
    # TBA add logic here
    imu_signals = None
    labels = None
    return imu_signals, labels



