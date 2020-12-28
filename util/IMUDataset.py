from skimage.io import imread
from torch.utils.data import Dataset
import pandas as pd
from os.path import join
import numpy as np
import logging


class IMUDataset(Dataset):
    """
        A class representing a dataset for IMU learning tasks
    """

    def __init__(self, imu_dataset_file, window_size, task_type, input_size,
                 window_shift=None, exclude_multi_class_windows=False):
        """
        :param imu_dataset_file: (str) a file with imu signals and their labels
        :param window_size (int): the window size to consider
        :param task_type (str): seq-to-seq or seq-to-one
        :param input_size (int): the input size (e.g. 6 for 6 IMU measurements)
        :param window_shift (int): the overlap between each window
        :paraam exclude_multi_class_windows (bool): whether to exclude multi-class windows
        :return: an instance of the class
        """
        super(IMUDataset, self).__init__()
        # Read the file
        if window_shift is None:
            window_shift = window_size
        df = pd.read_csv(imu_dataset_file)
        if df.shape[1] == 1:
            df = pd.read_csv(imu_dataset_file, delimiter='\t')
        # Fetch the flatten IMU data and labels
        flatten_imu = df.iloc[:, :input_size].values
        flatten_labels = df.iloc[:, input_size:].values
        n = flatten_labels.shape[0]
        assert n % window_size == 0
        imu = []
        labels = []

        # Collect labels and data per window
        n = flatten_imu.shape[0]
        start_index = 0
        while True:

            if start_index + window_size > n:
                break
            window_indices = list(range(start_index, (start_index + window_size)))

            add_sample = True
            label = flatten_labels[window_indices, :]
            if task_type == "seq-to-one":

                if len(np.unique(label)) > 1 and exclude_multi_class_windows:
                    logging.info("window excluded - more than one class present")
                    add_sample = False
                else:
                    label = label[0][0]

            if add_sample:
                imu.append(flatten_imu[window_indices, :])
                labels.append(label)
            start_index = start_index + window_shift
        self.labels = labels
        self.imu = imu
        logging.info("Dataset parsed - number of windows: {} (generated from {} non-overlapping windows) ".format(
            len(self.labels), n // window_size))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {'imu': self.imu[idx],
                  'label': self.labels[idx]}
        return sample



