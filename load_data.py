import numpy as np
import matplotlib.pyplot as plt
from transforms3d.euler import mat2euler

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
  function to read visual features, IMU measurements and calibration parameters
  Input:
      file_name: the input data file. Should look like "XXX_sync_KLT.npz"
  Output:
      t: time stamp
          with shape 1*t = (1,1106)
      features: visual feature point coordinates in stereo images, 
          with shape 4*n*t, where n is number of features = (4,n=206,feature_no=1106)
      linear_velocity: IMU measurements in IMU frame
          with shape 3*t = (3,1106)
      rotational_velocity: IMU measurements in IMU frame
          with shape 3*t = (3,1106)
      K: (left)camera intrinsic matrix
          [fx  0 cx
            0 fy cy
            0  0  1]
          with shape 3*3
      b: stereo camera baseline
          with shape 1
      cam_T_imu: extrinsic matrix from IMU to (left)camera optical frame, in SE(3).
          close to 
          [ 0 -1  0 t1
            0  0 -1 t2
            1  0  0 t3
            0  0  0  1]
          with shape 4*4
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''



def load_data(file_name):

  data_set = {}
  with np.load(file_name) as data:
      t = data["time_stamps"] # time_stamps
      data_set['t'] = t

      features = data["features"] # 4 x num_features : pixel coordinates of features
      data_set['features'] = features

      linear_velocity = data["linear_velocity"] # linear velocity measured in the body frame
      data_set['linear_velocity'] = linear_velocity

      rotational_velocity = data["rotational_velocity"] # rotational velocity measured in the body frame
      data_set['rotational_velocity'] = rotational_velocity

      K = data["K"] # intrinsic calibration matrix
      data_set['K'] = K

      b = data["b"] # baseline
      data_set['b'] = b

      cam_T_imu = data["cam_T_imu"] # Transformation from imu to camera frame
      data_set['cam_T_imu'] = cam_T_imu

  return data_set

street = 42 
filename = "data/00{}.npz".format(street)
data_set = load_data(filename)

def data_review(data_set,print_data=False, print_data_shape=True):
  for i in data_set.keys():
    if print_data:
      print('-----------'+i+'----------')
      print(data_set[i])
      print('\n')
    if print_data_shape:
      print('The shape of %s: %s' %(i, data_set[i].shape))
      print('\n')

data_review(data_set,print_data=True, print_data_shape=True)






