import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

number_of_samples = 2000
plot_on = False

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
# plt.subplot(2, 1, 1)
# plt.plot(gyro_vec_gt[:, 0], label='phi')
# plt.plot(gyro_vec_gt[:, 1], label='theta')
# plt.plot(gyro_vec_gt[:, 2], label='psy')
# plt.title('Omega')
# plt.xlabel('Num of samples')
# plt.ylabel('[rad/sec]')
# plt.legend()
# plt.grid()
#
# plt.subplot(2, 1, 2)
# plt.plot(gyro_vec_noised[:, 0], label='phi')
# plt.plot(gyro_vec_noised[:, 1], label='theta')
# plt.plot(gyro_vec_noised[:, 2], label='psy')
# plt.title('Omega Noised')
# plt.xlabel('Num of samples')
# plt.ylabel('[rad/sec]')
# plt.legend()
# plt.grid()
# if plot_on:
#     plt.show()


# Saving the data
# data_dir_path = 'C:\\masters\\git\\imu-transformers\\datasets\\'
# data_name = '21_12_20_static_gyro_2K.csv'
#
# concat_data = np.hstack((gyro_vec_noised, gyro_vec_gt))
# dff = pd.DataFrame(data=concat_data)
# dff.to_csv(data_dir_path + data_name, header=['gyro_x_noised', 'gyro_y_noised', 'gyro_z_noised','gyro_x', 'gyro_y', 'gyro_z'], index=False)

### Prepare acceleration vector - different angle sequence simlation
num_samples = 1000
phi = np.random.rand()*np.pi/2
print(phi)
theta = np.random.rand()*np.pi/2
print(theta)
g = 9.806
angles_vec = np.array([-np.sin(theta), np.sin(phi)*np.cos(theta), np.cos(phi)*np.cos(theta)])
print(angles_vec)
accel_vec = angles_vec*g
print(np.linalg.norm(accel_vec))
accel_seq = np.vstack([accel_vec]*num_samples)
print(accel_seq)


# plt.plot(accel_seq[:, 0], label='accel_x')
# plt.plot(accel_seq[:, 1], label='accel_y')
# plt.plot(accel_seq[:, 2], label='accel_z')
# plt.title('Omega')
# plt.xlabel('Num of samples')
# plt.ylabel('[m/sec^2]')
# plt.legend()
# plt.grid()
# plt.show()

def generate_accel_rand(num_samples):
    phi = np.random.rand() * np.pi/2
    theta = np.random.rand() * np.pi/2
    g = 9.806
    angles_vec = np.array([-np.sin(theta), np.sin(phi) * np.cos(theta), np.cos(phi) * np.cos(theta)])
    accel_vec = angles_vec * g
    accel_seq = np.vstack([accel_vec] * num_samples)
    return accel_seq


num_of_seq = 3
num_samples = 1000
accel_seq_all = generate_accel_rand(num_samples)

for i in range(num_of_seq):
    current_seq = generate_accel_rand(num_samples)
    accel_seq_all = np.vstack([accel_seq_all, current_seq])


b_f = (20*(1e-3))/9.81
# generating white noise
mu_wf = 0
sigma_wf = 500*0.2*(1e-2)/9.81
w_f = np.random.normal(mu_wf, sigma_wf, (len(accel_seq_all), 3))
f_noise = b_f + w_f
accel_seq_all_noised = accel_seq_all + f_noise  #noised


plt.subplot(2, 1, 1)
plt.plot(accel_seq_all[:, 0], label='accel_x')
plt.plot(accel_seq_all[:, 1], label='accel_y')
plt.plot(accel_seq_all[:, 2], label='accel_z')
plt.title('Acceleration')
plt.xlabel('Num of samples')
plt.ylabel('[m/sec^2]')
plt.legend()
plt.grid()


plt.subplot(2, 1, 2)
plt.plot(accel_seq_all_noised[:, 0], label='accel_x')
plt.plot(accel_seq_all_noised[:, 1], label='accel_y')
plt.plot(accel_seq_all_noised[:, 2], label='accel_z')
plt.title('Acceleration Noised')
plt.xlabel('Num of samples')
plt.ylabel('[m/sec^2]')
plt.legend()
plt.grid()
plt.show()