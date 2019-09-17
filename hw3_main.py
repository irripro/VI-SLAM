from EKF_slam import *

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
  Variable Setting
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# available street number : 27 & 42(train set) 20(test set)
#---------------- dataset loading and extracting ----------------------------------# 
street = 20
filename = "data/{:04d}.npz".format(street)
data_set = load_data(filename)

t = data_set['t'] # time_stamps
v_ = data_set['linear_velocity'][0,:] # linear_velocity
yaw_rate_ = data_set['rotational_velocity'][2,:] # yaw_rate
RFID_ = data_set['features']
K = data_set['K']

#---------------- color map setting for trainsets and testset ----------------------# 
color_map = {27:['palevioletred','Purple','darkslateblue'],
            42:['yellowgreen','orange','lightskyblue'],
            20:['steelblue','crimson','mediumvioletred']}

#----------------------- animation and plots saving set ----------------------------#
display_animation = True # demonstrate EKF localization over time by animation
plot_path = "plots/slam" # save path for plots over time saved
#save_extension = '.png'

#----------------------- hardware and state setting --------------------------------#
if street ==20:
	max_observation_range = 25.0  # maximum observation range
else:
	max_observation_range = 20.0
ma_dist_threshold = 2.0  # threshold of Mahalanobis distance for data association.
pose_size = 3  # robot pose size [x,y,yaw]
lm_pos_size = 2  # landmark position state size [x,y]

#----------------------------- EKF data saving set ---------------------------------# 
save_data = False # whether to save data after completing localization
save_dir = "saveData/slam_data/ekf_slam_data_{}".format(street)
video_record = False


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
  EKF Variance Setting
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# EKF state covariance
Cx = np.diag([0.5, 0.5, np.deg2rad(30.0)]) ** 2

#  Simulation parameter
Qsim = np.diag([0.2, np.deg2rad(1.0)]) ** 2
Rsim = np.diag([1.0, np.deg2rad(10.0)]) ** 2


def main():
	
	EKF = SLAM_EKF()

	for i in trange(t.shape[1]):
		if street == 20:
			if i in [815,820,825,830]:
				continue
		if i != (t.shape[1]-1):
			tau = t[0][i+1]-t[0][i]
		else:
			tau = t[0][i]-t[0][i-1] 
		v = v_[i]
		yaw_rate = yaw_rate_[i]

		RFID = RFID_[:,:,i]

		EKF.activate_ekf_slam(v,yaw_rate,RFID,tau)
		if display_animation:  # pragma: no cover
			flag = 0
			if not i%5 or i == t.shape[1]-1:
				flag = 1
				EKF.plot_slam_results(color_map,street,flag,i,plot_path,covariance_ellipse=True)

	if save_data == True:
		ekf_slam_data = EKF.get_params(param_set=['pst_truth','pst_est'],save_all=True)
		print('...start saving data...')
		with open(save_dir, 'wb') as slam_data:
			pickle.dump(ekf_slam_data,slam_data)
    
	if video_record == True:
		video_slam(street,save_extension)
	
	plt.show()

if __name__ == '__main__':
    main()