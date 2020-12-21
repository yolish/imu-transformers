import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

number_of_samples = 2000
plot_on = True

gyro_vec_gt = np.zeros((number_of_samples,3))

# adding noise to w_vec
# generating offset
b_w = np.pi/180
# generating random noise
mu_ww = 0
sigma_ww = 2*(0.03*10*np.pi)/180
w_w = np.random.normal(mu_ww, sigma_ww, (number_of_samples, 3))
w_noise = b_w + w_w
gyro_vec_noised = gyro_vec_gt + w_noise   #noised

#plotting
plt.subplot(2, 1, 1)
plt.plot(gyro_vec_gt[:, 0], label='phi')
plt.plot(gyro_vec_gt[:, 1], label='theta')
plt.plot(gyro_vec_gt[:, 2], label='psy')
plt.title('Omega')
plt.xlabel('Num of samples')
plt.ylabel('[rad/sec]')
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(gyro_vec_noised[:, 0], label='phi')
plt.plot(gyro_vec_noised[:, 1], label='theta')
plt.plot(gyro_vec_noised[:, 2], label='psy')
plt.title('Omega Noised')
plt.xlabel('Num of samples')
plt.ylabel('[rad/sec]')
plt.legend()
plt.grid()
if plot_on:
    plt.show()


# Saving the data
data_dir_path = 'C:\\masters\\git\\imu-transformers\\datasets\\'
data_name = '21_12_20_static_gyro_2K.csv'

concat_data = np.hstack((gyro_vec_noised, gyro_vec_gt))
dff = pd.DataFrame(data=concat_data)
dff.to_csv(data_dir_path + data_name, header=['gyro_x_noised', 'gyro_y_noised', 'gyro_z_noised','gyro_x', 'gyro_y', 'gyro_z'], index=False)
