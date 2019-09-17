import csv
import datetime
import time
import numpy as np
import os
import numpy as np
from scipy.linalg import expm, sinm, cosm
import matplotlib.pyplot as plt; plt.ion()
from numpy.linalg import inv
from transforms3d.euler import mat2euler
from mpl_toolkits import mplot3d


load_x_path = 'x_pos.npy'
load_y_path = 'y_pos.npy'
load_z_path = 'z_pos.npy'

xpos_list = np.load(load_x_path)
ypos_list = np.load(load_y_path)
zpos_list = np.load(load_z_path)

x_one_list = np.ones(len(xpos_list))
y_one_list = np.ones(len(ypos_list))
z_one_list = np.ones(len(zpos_list))

abs_xpos_list = xpos_list - x_one_list * xpos_list[0]
abs_ypos_list = ypos_list - y_one_list * ypos_list[0]
abs_zpos_list = zpos_list - z_one_list * zpos_list[0]

load_gps_timestamp_path = 'gps_timestamp.npy'
gps_timestamp = np.load(load_gps_timestamp_path)  #unit: unix time
time_one_list = np.ones(len(gps_timestamp))
abs_time_list = np.abs(np.array(gps_timestamp) - gps_timestamp[0] * time_one_list)
abs_time_list = [round(x, 3) for x in abs_time_list]

load_sat_timestamp_path = 'sat_time.npy'
sat_timestamp_str = np.load(load_sat_timestamp_path)  #unit: unix time
sat_time_list = []
for i in sat_timestamp_str:
    sat_timestamp = float(i[6:]) / 1000
    sat_time_list.append(sat_timestamp)

sat_time_one_list = np.ones(len(sat_time_list))
abs_sattime_list = np.abs(np.array(sat_time_list) - sat_time_list[0] * sat_time_one_list)

with open('7_12_LI_GROUP.csv') as csvfile1:
    readCSV = csv.reader(csvfile1, delimiter=',')
    index = 0
    vel = np.zeros([3, 1]).reshape(3, 1)      #unit m/s
    ang_vel = np.zeros([3,1]).reshape(3, 1)   #unit deg/s
    for row in readCSV:
        if row != []:
            index += 1
            cur_x_v = float(row[1])
            cur_y_v = float(row[2])
            cur_z_v = float(row[3])
            if row[4] != '':
                cur_x_w = float(row[4]) * np.pi / 180
                cur_y_w = float(row[5]) * np.pi / 180
                cur_z_w = float(row[6]) * np.pi / 180
            else:
                cur_x_w = 0
                cur_y_w = 0
                cur_z_w = 0
            cur_vel_column = np.array([cur_x_v, cur_y_v, cur_z_v]).reshape(3, 1)
            cur_ang_vel_column = np.array([cur_x_w, cur_y_w, cur_z_w]).reshape(3, 1)
            vel = np.concatenate((vel, cur_vel_column), axis=1)
            ang_vel = np.concatenate((ang_vel, cur_ang_vel_column), axis=1)
    vel = np.delete(vel, (0), axis=1)
    ang_vel = np.delete(ang_vel, (0), axis=1)

screw_axis = np.zeros((6, index))    # 6*1 matrix for linear velocity and angular velocity data
se3 = np.zeros((4, 4))
SE3 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
I = SE3
combine_skew_matrix = np.zeros((6, 6))
sigma = np.eye(6)
H_jacobian = np.array([[1, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0],
                       [0, 0, 0, 1, 0, 0]])

imu_xyz_info = np.zeros((2, 2)).reshape(4, 1)
world_T_imu = np.zeros((4, 4, index-1))
imu = np.zeros((4, 4, index-1))
x_car_trajectory = []
y_car_trajectory = []


for i in range(1, index-1):
    t = (sat_time_list[i] - sat_time_list[i - 1])
    W = 1
    V = 1
# mu prediction
    screw_axis[0:3, i] = vel[:, i]
    screw_axis[3:6, i] = ang_vel[:, i]
    skew_angular_matrix = np.array([[0, -ang_vel[2, i], ang_vel[1, i]],
                       [ang_vel[2, i], 0, -ang_vel[0, i]],
                       [-ang_vel[1, i], ang_vel[0, i], 0]])
    skew_angular_matrix_t = t * skew_angular_matrix
    # skew_angular_matrix_t = 0.0003 * skew_angular_matrix
    se3[0:3, 0:3] = skew_angular_matrix_t
    se3[0:3, 3] = vel[:, i] * t
    # se3[0:3, 3] = vel[:, i] * 0.0003
    SE3_current = expm(-se3)

    SE3 = SE3_current @ SE3
    I = SE3_current @ I
    imu[:, :, i] = np.linalg.inv(I)

# sigma prediction
    skew_vel_matrix = np.array([[0, -vel[2, i], vel[1, i]],
                       [vel[2, i], 0, -vel[0, i]],
                       [-vel[1, i], vel[0, i], 0]])
    combine_skew_matrix[0:3, 0:3] = skew_angular_matrix
    combine_skew_matrix[0:3, 3:6] = skew_vel_matrix
    combine_skew_matrix[3:6, 3:6] = skew_angular_matrix
    sigma_current = expm(-combine_skew_matrix * t)
    sigma = sigma_current * sigma * sigma_current.T + (t**2) * W * np.eye(6)

# Kalman gain
    imu_xyz_info[0:3]= np.linalg.inv(SE3)[0:3, 3].reshape(3, 1)
    # print(imu_xyz_info)
    gps_xyz_info = np.array([abs_xpos_list[i], abs_ypos_list[i], abs_zpos_list[i], 0]).reshape(4, 1) * 0.001
    inverse_imu_gps = np.linalg.pinv(H_jacobian @ sigma @ H_jacobian.T + V * np.eye(4))

    kalman_gain = sigma @ H_jacobian.T @ inverse_imu_gps
    print(kalman_gain)

# mu update
    diff_z_zhat = gps_xyz_info - imu_xyz_info
    # print(gps_xyz_info)
    # print(imu_xyz_info)
    # print(diff_z_zhat)
    kgain_times_diff = kalman_gain @ diff_z_zhat
    update_skew_matrix = np.array([[0, -kgain_times_diff[5], kgain_times_diff[4]],
                       [kgain_times_diff[5], 0, -kgain_times_diff[3]],
                       [-kgain_times_diff[4], kgain_times_diff[3], 0]])

    se3[0:3, 0:3] = update_skew_matrix
    se3[0, 3] = kgain_times_diff[0]
    se3[1, 3] = kgain_times_diff[1]
    se3[2, 3] = kgain_times_diff[2]
    se3[3, 3] = 0

    SE3 = expm(se3) @ SE3
# sigma update

    sigma = (np.eye((6)) - kalman_gain @ H_jacobian) @ sigma
    world_T_imu[:, :, i] = np.linalg.inv(SE3)


# fig3 = plt.figure()
# plt.scatter(x_car_trajectory, y_car_trajectory, s=5, alpha=1.0, label = 'Position')
# plt.draw()
# plt.waitforbuttonpress(0) # this will wait for indefinite time
# plt.close(fig3)
# plt.show()


def visualize_trajectory_2d(pose, show_ori=True):
  '''
  function to visualize the trajectory in 2D
  Input:
      pose:   4*4*N matrix representing the camera pose,
              where N is the number of pose, and each
              4*4 matrix is in SE(3)
  '''
  fig,ax = plt.subplots(figsize=(5,5))
  n_pose = pose.shape[2]
  ax.plot(pose[0,3,:],pose[1,3,:],'r-')
  ax.scatter(pose[0,3,0],pose[1,3,0],marker='s',label="start")
  ax.scatter(pose[0,3,-1],pose[1,3,-1],marker='o',label="end")
  # plt.scatter(landmarks_pos[0], landmarks_pos[1])
  if show_ori:
      select_ori_index = list(range(0,n_pose,int(n_pose/100)))
      yaw_list = []
      for i in select_ori_index:
          _,_,yaw = mat2euler(pose[:3,:3,i])
          yaw_list.append(yaw*900)
      dx = np.cos(yaw_list)
      dy = np.sin(yaw_list)
      dx,dy = [dx,dy]/np.sqrt(dx**2+dy**2)
      ax.quiver(pose[0,3,select_ori_index],pose[1,3,select_ori_index],dx,dy,\
          color="b",units="xy",width=0.5)
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.axis('equal')
  ax.grid(False)
  ax.legend()
  plt.show(block=True)
  return fig, ax

# 2-D plot
world_T_imu = world_T_imu * 1000
imu = imu * 1000
visualize_trajectory_2d(world_T_imu)
#
#
print(world_T_imu[:, :, 30])
print(imu[:, :, 30])

fig = plt.figure()
plt.scatter(abs_xpos_list, abs_ypos_list, s=5, alpha=1.0, label = 'GPS Position')
plt.scatter(imu[0, 3, :], imu[1, 3, :], s = 5, label = 'IMU Position')
plt.scatter(world_T_imu[0, 3, :], world_T_imu[1, 3, :], s = 5, label = 'EKF_IMU Position')

plt.scatter(abs_xpos_list[0], abs_ypos_list[0], s=50, color = 'red', label = 'Start')
plt.scatter(abs_xpos_list[-1], abs_ypos_list[-1], s=50, color = 'green', label = 'End')
plt.xlabel('Absolute X Position unit(m)')
plt.ylabel('Absolute Y Position unit(m)')
plt.title('X and Y Position Value from GPS Module')
plt.legend(loc=1)
plt.draw()
plt.waitforbuttonpress(0) # this will wait for indefinite time
plt.show()
# 3-D plot
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter3D(world_T_imu[0, 3, :], world_T_imu[1, 3, :], world_T_imu[2, 3, :])
# ax.scatter3D(abs_xpos_list, abs_ypos_list, abs_zpos_list)
# plt.draw()
# plt.waitforbuttonpress(0) # this will wait for indefinite time
# plt.show()
