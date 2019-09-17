'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
<EKF_Simultaneous Localization and Mapping>
	1. EKF localization
	2. EKF mapping 
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import math
import numpy as np
import matplotlib.pyplot as plt
import pdb
from load_data import *
from tqdm import trange
import pickle
from video_writer import video_slam
import os

from hw3_main import * # some global variables defined in the main file

fsu = K[0,0]
fsv = K[1,1]
cu = K[0,2]
cv = K[1,2]

class SLAM_EKF():

	'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
	I. Variable Initialization
	'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
	def __init__(self):
		self.u = np.array([[0,0]]).T # control input
		self.u_noise = np.array([[0,0]]).T 
		self.tau = 0 # time difference

		#  params for the current time stamp
		self.x_truth = np.zeros((pose_size,1)) # true values
		self.x_dead_rec = np.zeros((pose_size,1)) # dead reckoning 

		self.x_est = np.zeros((pose_size,1)) # estimated values
		self.cov_est = np.eye(pose_size) # estimated covariance

		# params that store past values for plotting results
		self.pst_est = np.zeros((pose_size,1))
		self.pst_truth = np.zeros((pose_size,1))
		self.pst_dead_rec = np.zeros((pose_size,1))
		#self.pst_z = np.zeros(self.z.shape)

	'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
	II. paramter update or parameter access
	'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

	def update_control_input(self,v,yaw_rate):
	    self.u = np.array([[v, yaw_rate]]).T

	def update_tau(self,tau):
		self.tau = tau

	def update_RFID(self,RFID):
		self.RFID = RFID

	def update_past_params(self):

		x_state = self.x_est[0:pose_size]

		# store data history
		self.pst_est = np.hstack((self.pst_est, x_state))
		self.pst_dead_rec = np.hstack((self.pst_dead_rec, self.x_dead_rec))
		self.pst_truth = np.hstack((self.pst_truth, self.x_truth))
		#self.pst_z = np.hstack((self.pst_z, self.z))

	def get_params(self,param_set=['pst_truth','pst_est'],save_all=True):
		assert isinstance(param_set,list)
		data = {'u':self.u,'tau':self.tau,'x_dead_rec':self.x_dead_rec,
				'x_truth':self.x_truth,'x_est':self.x_est,'pst_truth':self.pst_truth,
				'pst_est':self.pst_est,'cov_est':self.cov_est,
				'u_noise':self.u_noise,'pst_dead_rec':self.pst_dead_rec,'z':self.z}
		for i in param_set:
			assert isinstance(i,str) and i in data.keys()
		opt_data = {}
		if save_all == True:
			opt_data = data
		else:
			for i in param_set:
				opt_data[i] = data[i]
		return opt_data

	'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
	III. Motion Model and Jacobian
	'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

	def motion_model(self,x, u):
		'''
		: process pose with motion model
		'''
		F = np.array([[1.0, 0, 0],
					 [0, 1.0, 0],
					 [0, 0, 1.0]])
		B = np.array([[self.tau * math.cos(x[2, 0]), 0],
					 [self.tau * math.sin(x[2, 0]), 0],
					 [0.0, self.tau]])
		x = (F @ x) + (B @ u)
		return x

	def jacobian_motion(self,x, u):
		'''
		: outputs:
			1. param G: 
			   type G: np.array, shape()
			2. param Fx: jacobian of motion model
			   type Fx: np.array, shape()

		x_{t+1} = x_t + v * DT * cos(yaw)
		y_{t+1} = y_t + v * DT * sin(yaw)
		v_{t+1} = v_t
		yaw_{t+1} = yaw_t + omega * DT

		... then derivatives will be ...
		dx / dyaw = -v * DT * sin(yaw)
		dx / dv = DT * cos(yaw)
		dy / dyaw = v * DT * cos(yaw)
		dy / dv = DT * sin(yaw)

		'''
		Fx = np.hstack((np.eye(pose_size), np.zeros(
			(pose_size, lm_pos_size * self.get_LM_no(x)))))

		jF = np.array([[0.0, 0.0, -self.tau * u[0] * math.sin(x[2, 0])],
					  [0.0, 0.0, self.tau * u[0] * math.cos(x[2, 0])],
					  [0.0, 0.0, 0.0]])

		G = np.eye(pose_size) + Fx.T * jF * Fx

		return G, Fx,
	'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
	IV. Observation Model and Jacobian
	'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

	def observation(self):

	    self.x_truth = self.motion_model(self.x_truth, self.u)

	    # add noise to gps x-y
	    self.z = np.zeros((0, pose_size)) # observation vector
	    #self.pst_z = np.zeros(self.z.shape)

	    for i in range(len(self.RFID[:, 0])):

	        dx = self.RFID[i, 0] - self.x_truth[0, 0]
	        dy = self.RFID[i, 1] - self.x_truth[1, 0]
	        d = math.sqrt(dx**2 + dy**2)
	        angle = self.pi_2_pi(math.atan2(dy, dx) - self.x_truth[2, 0])

	        if d <= max_observation_range:
	        	# distance noise
	            dn = d + np.random.randn() * Qsim[0, 0]  
	            # angle noise
	            anglen = angle + np.random.randn() * Qsim[1, 1]  
	            
	            zi = np.array([dn, anglen, i])
	            self.z = np.vstack((self.z, zi))

	    # add noise to input
	    self.u_noise = np.array([[
	        self.u[0, 0] + np.random.randn() * Rsim[0, 0],
	        self.u[1, 0] + np.random.randn() * Rsim[1, 1]]]).T

	    self.x_dead_rec = self.motion_model(self.x_dead_rec, self.u_noise)

	def jacobian_H(self,q, delta, x, i):
	    # sqaure root of q
	    sq = math.sqrt(q)
	    G = np.array([[-sq * delta[0, 0], - sq * delta[1, 0], 0, sq * delta[0, 0], sq * delta[1, 0]],
	                  [delta[1, 0], - delta[0, 0], - 1.0, - delta[1, 0], delta[0, 0]]])

	    # derivative of projection function
	    G = G / q
	    nLM = self.get_LM_no(x)
	    F1 = np.hstack((np.eye(3), np.zeros((3, 2 * nLM))))
	    F2 = np.hstack((np.zeros((2, 3)), np.zeros((2, 2 * (i - 1))),
	                    np.eye(2), np.zeros((2, 2 * nLM - 2 * i))))

	    F = np.vstack((F1, F2))

	    H = G @ F

	    return H

	'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
	V. EKF mapping:
		1. get landmark pose based on observation vectors
		2. get landmark pose based on state vector
		3. landmark association
		4. get landmark number 
		5. back projection
	'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

	def get_LM_Pos(self,x, z):
		'''
		: compute landmark pose based on observations
		'''
		zp = np.zeros((2, 1))

		zp[0, 0] = x[0, 0] + z[0] * math.cos(x[2, 0] + z[1])
		zp[1, 0] = x[1, 0] + z[0] * math.sin(x[2, 0] + z[1])

		return zp


	def get_LM_Pos_from_state(self,x, ind):
		'''
		: compute landmark pose from state vector
		'''

		lm = x[pose_size + lm_pos_size * ind: pose_size + lm_pos_size * (ind + 1), :]
		return lm


	def get_corresponding_LM_ID(self,xAug, PAug, zi):
	    '''
	    : landmark association 
	    '''

	    nLM = self.get_LM_no(xAug)

	    mdist = []

	    for i in range(nLM):
	        lm = self.get_LM_Pos_from_state(xAug, i)
	        y, S, H = self.back_projection(lm, xAug, PAug, zi, i)
	        mdist.append(y.T @ np.linalg.inv(S) @ y)

	    mdist.append(ma_dist_threshold)  # new landmark

	    minid = mdist.index(min(mdist))

	    return minid

	def get_LM_no(self,x):
		'''
		: get landmark number
		'''
		n = int((len(x) - pose_size) / lm_pos_size)
		return n

	def back_projection(self,lm, xEst, PEst, z, LMid):
		delta = lm - xEst[0:2]
		q = (delta.T @ delta)[0, 0]
		zangle = math.atan2(delta[1, 0], delta[0, 0]) - xEst[2, 0]
		zp = np.array([[math.sqrt(q), self.pi_2_pi(zangle)]])
		y = (z - zp).T
		y[1] = self.pi_2_pi(y[1])
		H = self.jacobian_H(q, delta, xEst, LMid + 1)
		# H_t * Sigama_t * H_t^T + I@V
		S = H @ PEst @ H.T + Cx[0:2, 0:2]

		return y, S, H

	def pi_2_pi(self,angle):
		'''
		: process the angle to ensure it's in [-pi,pi] range

		'''
		return (angle + math.pi) % (2 * math.pi) - math.pi

	'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
	VI. EKF Implementation
	'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

	def ekf_slam(self):

	    # predict
	    S = pose_size
	    self.x_est[0:S] = self.motion_model(self.x_est[0:S],self.u_noise)

	    G, Fx = self.jacobian_motion(self.x_est[0:S], self.u_noise)
	    self.cov_est[0:S, 0:S] = G.T * self.cov_est[0:S, 0:S] * G + Fx.T * Cx * Fx
	    
	    initP = np.eye(2)

	    # update
	    for iz in range(len(self.z[:, 0])):  # for each observation
	        minid = self.get_corresponding_LM_ID(self.x_est, self.cov_est, self.z[iz, 0:2])

	        nLM = self.get_LM_no(self.x_est)
	        if minid == nLM:
	            print('...Get New Landmark ...')

	            # extend state and covariance matrix
	            xAug = np.vstack((self.x_est, self.get_LM_Pos(self.x_est, self.z[iz, :])))
	            PAug = np.vstack((np.hstack((self.cov_est, np.zeros((len(self.x_est), lm_pos_size)))),
	                              np.hstack((np.zeros((lm_pos_size, len(self.x_est))), initP))))
	            self.x_est = xAug
	            self.cov_est = PAug

	        lm = self.get_LM_Pos_from_state(self.x_est, minid)
	        y, S, H = self.back_projection(lm, self.x_est,self.cov_est, self.z[iz, 0:2], minid)

	        # EKF update
	        # get K_t
	        K = (self.cov_est @ H.T) @ np.linalg.inv(S)
	        # get mu_t+1
	        self.x_est = self.x_est + (K @ y)
	        # get Sigama_t+1
	        self.cov_est = (np.eye(len(self.x_est)) - (K @ H)) @ self.cov_est

	    self.x_est[2] = self.pi_2_pi(self.x_est[2])

	def activate_ekf_slam(self,v,yaw_rate,RFID,tau):
		'''
		: avtivate EKF localiztaion with params obtained from different time stamps
		'''

		self.update_control_input(v,yaw_rate)
		self.update_tau(tau)
		self.update_RFID(RFID)
		self.observation()
		self.ekf_slam()
		self.update_past_params()

	'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
	VI. EKF SLAM Resulting Plots 
	'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

	def plot_covariance_ellipse(self,xEst, PEst):
		'''
		: to plot the covariance ellipse at the current localization point
		'''
		Pxy = PEst[0:2, 0:2]
		eigval, eigvec = np.linalg.eig(Pxy)

		if eigval[0] >= eigval[1]:
			bigind = 0
			smallind = 1
		else:
			bigind = 1
			smallind = 0

		t = np.arange(0, 2 * math.pi + 0.1, 0.1)
		a = math.sqrt(eigval[bigind])
		b = math.sqrt(eigval[smallind])
		x = [a * math.cos(it) for it in t]
		y = [b * math.sin(it) for it in t]
		angle = math.atan2(eigvec[bigind, 1], eigvec[bigind, 0])
		R = np.array([[math.cos(angle), math.sin(angle)],
			[-math.sin(angle), math.cos(angle)]])
		fx = R @ (np.array([x, y]))
		px = np.array(fx[0, :] + xEst[0, 0]).flatten()
		py = np.array(fx[1, :] + xEst[1, 0]).flatten()
		return plt.plot(px, py, '--r')

	def plot_slam_results(self,color_map,street,flag,it,plt_dir,covariance_ellipse=True):

		plt.cla()
		
		# features
		p2 = None 
		#p1, = plt.plot(self.RFID[0, :], self.RFID[1, :], marker='*',c='#02d8e9')
		p1, = plt.plot(self.RFID[:, 0], self.RFID[:, 1], marker='*',c='#02d8e9')
		
		# estimated values 
		# plt.plot(self.x_est[0], self.x_est[1], '.r')

		# plot landmarks
		for j in range(self.get_LM_no(self.x_est)):
			p2, = plt.plot(self.x_est[pose_size + j * 2],
				self.x_est[pose_size + j * 2 + 1], marker='x',c=color_map[street][0])

		p3, = plt.plot(self.pst_truth[0, :],
				self.pst_truth[1, :], c=color_map[street][1])
		#plt.plot(self.pst_dead_rec[0, :],
		#		self.pst_dead_rec[1, :], '-k')
		p4, = plt.plot(self.pst_est[0, :].flatten(),
				self.pst_est[1, :].flatten(), c=color_map[street][2])

		# start point and end point
		p5 = plt.scatter(self.pst_truth[0, 0],self.pst_truth[1, 0],marker='s')
		p6 = plt.scatter(self.pst_truth[0, -1],self.pst_truth[1, -1],marker='o')

		# whether to plot the covariance ellipse at the current position 
		if covariance_ellipse == True:
			p7, = self.plot_covariance_ellipse(self.x_est, self.cov_est)
		
		if p2 is not None:
			plt.legend([p1,p2,p3,p4,p5,p6,p7], ['Estimated Features', 'Landmark','True Trajectory', 'Estimated Trajectory','Start','End','Covariance Ellipse'])
		else:
			plt.legend([p1,p3,p4,p5,p6,p7], ['Estimated Features','True Trajectory', 'Estimated Trajectory','Start','End','Covariance Ellipse'])

		plt.title(str(street)+' street: EKF SLAM')

		if flag:
			main_folder = "{}/{}".format(plot_path, street)
			if not os.path.exists(main_folder):
				os.mkdir(main_folder)
			plt.savefig("{}/{}.png".format(main_folder, it))

		plt.axis('equal')
		plt.grid(True)
		plt.pause(0.001)


#do_ekf_slam()

