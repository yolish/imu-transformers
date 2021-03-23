import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create a function that reads the raw txt file of the MRU recording and saves only the relevant data in csv format
mru_raw = pd.read_csv('C:\\Users\\a00497000\\Documents\\Personal\\Masters\\sem2\\nav_proj\\Data\\Static\\set1\\M2000279-2021-02-22-10-57-45.txt')
mru_raw = mru_raw[24:]
with open('test.txt', 'a') as f:
    f.write(
        mru_raw.to_string(header=False, index=False)
    )
mru_valid = pd.read_fwf('test.txt')



p40_data = pd.read_csv('C:\\Users\\a00497000\\Documents\\Personal\\Masters\\sem2\\nav_proj\\Data\\Static\\set1\\indoor_output_2021-02-22_10_59_35.csv')
a = 5

plt.subplot(2, 1, 1)
# plt.plot(mru_valid['Acc_X'], label='X')
# plt.plot(mru_valid['Acc_Y'], label='Y')
# plt.plot(mru_valid['Acc_Z'], label='Z')
plt.plot(mru_valid['Gyro_X'], label='X')
plt.plot(mru_valid['Gyro_Y'], label='Y')
plt.plot(mru_valid['Gyro_Z'], label='Z')

plt.title('MRU Device')

plt.subplot(2, 1, 2)
# plt.plot(p40_data['accX'], label='X')
# plt.plot(p40_data['accY'], label='Y')
# plt.plot(p40_data['accZ'], label='Z')
plt.plot(p40_data['gyroX'], label='X')
plt.plot(p40_data['gyroY'], label='Y')
plt.plot(p40_data['gyroZ'], label='Z')
plt.title('P40 Smartphone')
plt.show()