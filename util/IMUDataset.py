from skimage.io import imread
from torch.utils.data import Dataset
import pandas as pd
from os.path import join
import numpy as np


class IMUDataset(Dataset):
    """
        A class representing a dataset for IMU learning tasks
    """

    def __init__(self, imu_dataset_file, window_size, task_type):
        """
        :param imu_dataset_file: (str) a file with imu signals and their labels
        :param window_size (int): the window size to consider
        :param task_type (str): seq-to-seq or seq-to-one
        :return: an instance of the class
        """
        super(IMUDataset, self).__init__()
        # Read the file
        df = pd.read_csv(imu_dataset_file)
        # Fetch the flatten IMU data and labels
        flatten_imu = df.iloc[:, :6].values
        flatten_labels = df.iloc[:, 6:].values
        n = flatten_labels.shape[0]
        assert n % window_size == 0
        imu = []
        labels = []

        # Collect labels and data per window
        for i in range(0, n//window_size):
            imu.append(flatten_imu[i*window_size:(i*window_size+window_size), :])
            if task_type == "seq-to-seq":
                label = flatten_labels[i * window_size:(i * window_size + window_size), :]
            else: # seq-to-one
                label = flatten_labels[i * window_size, 0] # take class from first sample
            labels.append(label)

        self.labels = labels
        self.imu = imu

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {'imu': self.imu[idx],
                  'label': self.labels[idx]}
        return sample



