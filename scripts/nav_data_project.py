import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
# SET 1
# Create a function that reads the raw txt file of the MRU recording and saves only the relevant data in csv format
# mru_raw = pd.read_csv('C:\\Users\\a00497000\\Documents\\Personal\\Masters\\sem2\\nav_proj\\Data\\Static\\set1\\M2000279-2021-02-22-10-57-45.txt')
# mru_raw = mru_raw[24:]
# with open('test.txt', 'a') as f:
#     f.write(
#         mru_raw.to_string(header=False, index=False)
#     )
# mru_valid = pd.read_fwf('test.txt')
# f.close()
# os.remove('test.txt')
# mru_valid = mru_valid[300:14125]
# accell_gyro_mru = mru_valid[['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']]
# accell_gyro_mru.to_csv('set1_mru.csv', header=['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z'], index=False)
#
# p40_data = pd.read_csv('C:\\Users\\a00497000\\Documents\\Personal\\Masters\\sem2\\nav_proj\\Data\\Static\\set1\\indoor_output_2021-02-22_10_59_35.csv')
# p40_data = p40_data[1800:-3500]
# accell_gyro_p40 = p40_data[['accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ']]
# accell_gyro_p40.to_csv('set1_p40.csv', header=['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z'], index=False)


# #Set3
# mru_raw = pd.read_csv('C:\\Users\\a00497000\\Documents\\Personal\\Masters\\sem2\\nav_proj\\Data\\Static\\set3\\M2000279-2021-03-14-11-49-37.txt')
# mru_raw = mru_raw[24:]
# with open('test.txt', 'a') as f:
#     f.write(
#         mru_raw.to_string(header=False, index=False)
#     )
# f.close()
# mru_valid = pd.read_fwf('test.txt')
# os.remove('test.txt')
# mru_valid = mru_valid[2000:12000]
# accell_gyro_mru = mru_valid[['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']]
# accell_gyro_mru.to_csv('set3_mru.csv', header=['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z'], index=False)
#
# p40_data = pd.read_csv('C:\\Users\\a00497000\\Documents\\Personal\\Masters\\sem2\\nav_proj\\Data\\Static\\set3\\outdoor_output_2021-03-14_11_52_09.csv')
# p40_data = p40_data[500:20500]
# accell_gyro_p40 = p40_data[['accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ']]
# accell_gyro_p40.to_csv('set3_p40.csv', header=['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z'], index=False)


# SET 2 big 1
# mru_raw = pd.read_csv('C:\\Users\\a00497000\\Documents\\Personal\\Masters\\sem2\\nav_proj\\Data\\Static\\set2_big\\M2000279-2021-02-24-17-15-10.txt')
# mru_raw = pd.read_csv('C:\\Users\\a00497000\\Documents\\Personal\\Masters\\sem2\\nav_proj\\Data\\Static\\set2_big\\M2000279-2021-02-24-17-17-18.txt')
# mru_raw = pd.read_csv('C:\\Users\\a00497000\\Documents\\Personal\\Masters\\sem2\\nav_proj\\Data\\Static\\set2_big\\M2000279-2021-02-24-17-20-51.txt')
# mru_raw = pd.read_csv('C:\\Users\\a00497000\\Documents\\Personal\\Masters\\sem2\\nav_proj\\Data\\Static\\set2_big\\M2000279-2021-02-24-17-22-52.txt')
# mru_raw = pd.read_csv('C:\\Users\\a00497000\\Documents\\Personal\\Masters\\sem2\\nav_proj\\Data\\Static\\set2_big\\M2000279-2021-02-24-17-24-25.txt')
# mru_raw = pd.read_csv('C:\\Users\\a00497000\\Documents\\Personal\\Masters\\sem2\\nav_proj\\Data\\Static\\set2_big\\M2000279-2021-02-24-17-25-30_longrun.txt')
#
# mru_raw = mru_raw[24:]
# with open('test.txt', 'a') as f:
#     f.write(
#         mru_raw.to_string(header=False, index=False)
#     )
# f.close()
# mru_valid = pd.read_fwf('test.txt',index=False)
# os.remove('test.txt')
# mru_valid = mru_valid[0:10000]
# accell_gyro_mru = mru_valid[['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']]
# accell_gyro_mru.to_csv('set_2_f_mru.csv', header=['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z'], index=False)

# p40_data = pd.read_csv('C:\\Users\\a00497000\\Documents\\Personal\\Masters\\sem2\\nav_proj\\Data\\Static\\set2_big\\outdoor_output_2021-02-24_17_16_35.csv')
# p40_data = pd.read_csv('C:\\Users\\a00497000\\Documents\\Personal\\Masters\\sem2\\nav_proj\\Data\\Static\\set2_big\\outdoor_output_2021-02-24_17_19_26.csv')
# p40_data = pd.read_csv('C:\\Users\\a00497000\\Documents\\Personal\\Masters\\sem2\\nav_proj\\Data\\Static\\set2_big\\outdoor_output_2021-02-24_17_21_46.csv')
# p40_data = pd.read_csv('C:\\Users\\a00497000\\Documents\\Personal\\Masters\\sem2\\nav_proj\\Data\\Static\\set2_big\\outdoor_output_2021-02-24_17_23_58.csv')
# p40_data = pd.read_csv('C:\\Users\\a00497000\\Documents\\Personal\\Masters\\sem2\\nav_proj\\Data\\Static\\set2_big\\outdoor_output_2021-02-24_17_25_09.csv')
# p40_data = pd.read_csv('C:\\Users\\a00497000\\Documents\\Personal\\Masters\\sem2\\nav_proj\\Data\\Static\\set2_big\\outdoor_output_2021-02-24_17_27_15.csv')
#
#
# p40_data = p40_data[0:20000]
# accell_gyro_p40 = p40_data[['accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ']]
# accell_gyro_p40.to_csv('set_2_f_p40.csv', header=['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z'], index=False)


#set 4 big
# mru_raw = pd.read_csv('C:\\Users\\a00497000\\Documents\\Personal\\Masters\\sem2\\nav_proj\\Data\\Static\\set4_big\\M2000279-2021-03-17-17-18-16.txt')
# mru_raw = pd.read_csv('C:\\Users\\a00497000\\Documents\\Personal\\Masters\\sem2\\nav_proj\\Data\\Static\\set4_big\\M2000279-2021-03-17-17-23-12.txt')
# mru_raw = pd.read_csv('C:\\Users\\a00497000\\Documents\\Personal\\Masters\\sem2\\nav_proj\\Data\\Static\\set4_big\\M2000279-2021-03-17-17-27-17.txt')
# mru_raw = pd.read_csv('C:\\Users\\a00497000\\Documents\\Personal\\Masters\\sem2\\nav_proj\\Data\\Static\\set4_big\\M2000279-2021-03-17-17-32-25.txt')
# mru_raw = pd.read_csv('C:\\Users\\a00497000\\Documents\\Personal\\Masters\\sem2\\nav_proj\\Data\\Static\\set4_big\\M2000279-2021-03-17-17-38-09.txt')
mru_raw = pd.read_csv('C:\\Users\\a00497000\\Documents\\Personal\\Masters\\sem2\\nav_proj\\Data\\Static\\set4_big\\M2000279-2021-03-17-17-41-53.txt')
# mru_raw = pd.read_csv('C:\\Users\\a00497000\\Documents\\Personal\\Masters\\sem2\\nav_proj\\Data\\Static\\set4_big\\M2000279-2021-03-17-17-47-46.txt')



mru_raw = mru_raw[24:]
with open('test.txt', 'a') as f:
    f.write(
        mru_raw.to_string(header=False, index=False)
    )
f.close()
mru_valid = pd.read_fwf('test.txt')
os.remove('test.txt')
# mru_valid = mru_valid[2000:12000]
accell_gyro_mru = mru_valid[['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']]
accell_gyro_mru.to_csv('set_4_a_mru.csv', header=['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z'], index=False)

# p40_data = pd.read_csv('C:\\Users\\a00497000\\Documents\\Personal\\Masters\\sem2\\nav_proj\\Data\\Static\\set4_big\\indoor_output_2021-03-17_17_22_26.csv')
# p40_data = pd.read_csv('C:\\Users\\a00497000\\Documents\\Personal\\Masters\\sem2\\nav_proj\\Data\\Static\\set4_big\\indoor_output_2021-03-17_17_26_05.csv')
# p40_data = pd.read_csv('C:\\Users\\a00497000\\Documents\\Personal\\Masters\\sem2\\nav_proj\\Data\\Static\\set4_big\\indoor_output_2021-03-17_17_30_08.csv')
# p40_data = pd.read_csv('C:\\Users\\a00497000\\Documents\\Personal\\Masters\\sem2\\nav_proj\\Data\\Static\\set4_big\\indoor_output_2021-03-17_17_36_12.csv')
# p40_data = pd.read_csv('C:\\Users\\a00497000\\Documents\\Personal\\Masters\\sem2\\nav_proj\\Data\\Static\\set4_big\\indoor_output_2021-03-17_17_41_00.csv')
p40_data = pd.read_csv('C:\\Users\\a00497000\\Documents\\Personal\\Masters\\sem2\\nav_proj\\Data\\Static\\set4_big\\indoor_output_2021-03-17_17_44_17.csv')
# p40_data = pd.read_csv('C:\\Users\\a00497000\\Documents\\Personal\\Masters\\sem2\\nav_proj\\Data\\Static\\set4_big\\indoor_output_2021-03-17_17_51_06.csv')


# p40_data = p40_data[500:20500]
accell_gyro_p40 = p40_data[['accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ']]
accell_gyro_p40.to_csv('set_4_a_p40.csv', header=['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z'], index=False)







# Plotting
plt.subplot(2, 1, 1)
plt.plot(range(len(mru_valid)), mru_valid['Acc_X'], label='Acc_X')
plt.plot(range(len(mru_valid)), mru_valid['Acc_Y'], label='Acc_Y')
plt.plot(range(len(mru_valid)), mru_valid['Acc_Z'], label='Acc_Z')
plt.plot(range(len(mru_valid)), mru_valid['Gyro_X'], label='Gyro_X')
plt.plot(range(len(mru_valid)), mru_valid['Gyro_Y'], label='Gyro_Y')
plt.plot(range(len(mru_valid)), mru_valid['Gyro_Z'], label='Gyro_Z')
plt.legend()
plt.ylabel('IMU Signals Amp.')
plt.title('MRU Device')

plt.subplot(2, 1, 2)
plt.plot(range(len(p40_data)), p40_data['accX'], label='Acc_X')
plt.plot(range(len(p40_data)), p40_data['accY'], label='Acc_Y')
plt.plot(range(len(p40_data)), p40_data['accZ'], label='Acc_Z')
plt.plot(range(len(p40_data)), p40_data['gyroX'], label='Gyro_X')
plt.plot(range(len(p40_data)), p40_data['gyroY'], label='Gyro_Y')
plt.plot(range(len(p40_data)), p40_data['gyroZ'], label='Gyro_Z')
plt.legend()
plt.xlabel('Samples')
plt.ylabel('IMU Signals Amp.')
plt.title('P40 Smartphone')
plt.show()
