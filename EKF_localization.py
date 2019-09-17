import math
import numpy as np
import matplotlib.pyplot as plt
import pdb
from load_data import *
from tqdm import trange
import pickle
import os
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
  Variable Setting
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# available street number : 27 & 42(train set) 20(test set)
street = 42
filename = "/Users/momolee/Documents/PROJECTS/VI-SLAM/data/{:04d}.npz".format(street)
if street not in [20, 27, 42]:
    raise Exception('Invalid Street Number')
data_set = load_data(filename)

#---------------- color map setting for trainsets and testset ----------------------------------# 
color_map = {27:['palevioletred','Purple','darkslateblue'],
            42:['yellowgreen','orange','lightskyblue'],
            20:['steelblue','crimson','mediumvioletred']}

t = data_set['t'] # time_stamps
v_ = data_set['linear_velocity'][0,:] # linear_velocity
yaw_rate_ = data_set['rotational_velocity'][2,:] # yaw_rate

#----------------------- animation and plots saving set ----------------------------------#
display_animation = True # demonstrate EKF localization over time by animation
save_animation = True
plot_path = "/Users/momolee/Documents/PROJECTS/VI-SLAM/plots/localization"
ani_path = "/Users/momolee/Documents/PROJECTS/VI-SLAM/animation"

#----------------------------- EKF data saving set ----------------------------------# 
save_data = False # whether to save data after completing localization
save_dir = "/Users/momolee/Documents/PROJECTS/VI-SLAM/saveData/localization_data"

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
  EKF Variance Setting
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# predicted variance from EKF 
# the closer these are to the true values, the better the EKF performs
Q = np.diag([0.1, 0.1, np.deg2rad(1.0), 1.0])**2
R = np.diag([1.0, 1.0])**2

# actual variance for the EKF implementation
R_sim = np.diag([0.5, 0.5])**2
Q_sim = np.diag([1.0, np.deg2rad(30.0)])**2


class ekf_localization():
    '''
    : for implementing EKF localization

    '''

    def __init__(self):

        self.u = np.array([[0,0]]).T # control input
        self.u_noise = np.array([[0,0]]).T 
        self.tau = 0 # time difference
        self.z = np.zeros((2,1)) # observation vector
        
        #  params for the current time stamp
        self.x_truth = np.zeros((4,1)) # true values
        self.x_dead_rec = np.zeros((4,1)) # dead reckoning 
      
        self.x_est = np.zeros((4,1)) # estimated values
        self.cov_est = np.eye(4) # estimated covariance

        # params that store past values for plotting results
        self.pst_est = np.zeros((4,1))
        self.pst_truth = np.zeros((4,1))
        self.pst_dead_rec = np.zeros((4,1))
        self.pst_z = np.zeros((2, 1))
    
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    II. paramter update or parameter access
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    def update_tau(self,tau):
        self.tau = tau

    def update_control_input(self,v,yaw_rate):
        self.u = np.array([[v, yaw_rate]]).T
        #return self.u

    def update_x_truth(self):
        x_ = self.motion_model()
        self.x_truth = x_

    def update_x_dead_rec(self):
        x_ = self.noise_motion_model(self.x_dead_rec)
        self.x_dead_rec = x_

    def update_past_params(self):
        self.pst_est = np.hstack((self.pst_est, self.x_est))
        self.pst_dead_rec = np.hstack((self.pst_dead_rec, self.x_dead_rec))
        self.pst_truth = np.hstack((self.pst_truth, self.x_truth))
        self.pst_z = np.hstack((self.pst_z, self.z))

    def get_params(self,param_set=['pst_z','pst_Truth','pst_estimate'],save_all=True):
        assert isinstance(param_set,list)
        data = {'u':self.u,'tau':self.tau,'x_dead_rec':self.x_dead_rec,
                                                'x_truth':self.x_truth,'x_est':self.x_est,'pst_z':self.pst_z,
                                                'pst_Truth':self.pst_truth,'pst_estimate':self.pst_est,'cov_est':self.cov_est,
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

    def motion_model(self):
        F = np.array([
            [1.0, 0, 0, 0],
            [0, 1.0, 0, 0],
            [0, 0, 1.0, 0],
            [0, 0, 0, 0]
        ])
        B = np.array([
            [self.tau*math.cos(self.x_truth[2,0]), 0],
            [self.tau*math.sin(self.x_truth[2,0]), 0],
            [0.0, self.tau],
            [1.0, 0.0]
        ])
        return F.dot(self.x_truth) + B.dot(self.u)

    def jacobian_F(self,x):
        '''
        : compute Jacobian of motion model
        : motion model
            
            x_{t+1} = x_t+v*tau*cos(yaw)
            y_{t+1} = y_t+v*tau*sin(yaw)
            yaw_{t+1} = yaw_t+omega*tau
            v_{t+1} = v{t}
            
            ... so ...
            
            dx/dyaw = -v*tau*sin(yaw)
            dx/dv = tau*cos(yaw)
            dy/dyaw = v*tau*cos(yaw)
            dy/dv = tau*sin(yaw)
        '''
        yaw = x[2, 0]
        v = self.u_noise[0, 0]
        jF = np.array([
            [1.0, 0.0, -self.tau * v * math.sin(yaw), self.tau * math.cos(yaw)],
            [0.0, 1.0, self.tau * v * math.cos(yaw), self.tau * math.sin(yaw)],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]])
        return jF

    def noise_motion_model(self,x):
        F = np.array([
            [1.0, 0, 0, 0],
            [0, 1.0, 0, 0],
            [0, 0, 1.0, 0],
            [0, 0, 0, 0]
        ])
        B = np.array([
            [self.tau*math.cos(x[2,0]), 0],
            [self.tau*math.sin(x[2,0]), 0],
            [0.0, self.tau],
            [1.0, 0.0]
        ])
        return F.dot(x) + B.dot(self.u_noise)

    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    IV. Observation Model and Jacobianii
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    def observation(self):
        # get observation = GPS values + noise
        self.z = self.observation_model(self.x_truth) + R_sim.dot(np.random.randn(2,1))
        # add nosie to control input 
        self.u_noise = self.u + Q_sim.dot(np.random.randn(2,1))

        #return z, u_noisy

    def observation_model(self,x):
        H = np.array([
            [1.0, 0, 0, 0],
            [0, 1.0, 0, 0],
        ])
        return H.dot(x)

    def jacobian_H(self,x):
        '''
        : compute Jacobian of observation model
        '''
        jH = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        return jH

    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    V. EKF Implementation
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    def ekf(self):
        '''
        : implement EKF localization
        '''

        # conduct prediction
        x_pred = self.noise_motion_model(self.x_est)
        J_F = self.jacobian_F(x_pred)
        cov_pred = J_F@self.cov_est@J_F.T + Q

        # conduct update 
        J_H = self.jacobian_H(x_pred)
        z_est = self.observation_model(x_pred)
        y = self.z - z_est

        S = J_H@cov_pred@J_H.T + R
        K = cov_pred@J_H.T@np.linalg.inv(S)

        self.x_est = x_pred + K@y
        self.cov_est = (np.eye(len(self.x_est)) - K@J_H) @ cov_pred

        #return selfx_est, cov_est

    def activate_ekf_localization(self,v,yaw_rate,tau):
        '''
        : avtivate EKF localiztaion with params obtained from different time stamps
        '''

        # update u/control input and tau/time difference
        self.update_control_input(v,yaw_rate)
        self.update_tau(tau)
        # get the true value 
        self.update_x_truth()
        
        # update observation and control input with noise
        self.observation()
        # compute dead reckoning 
        self.update_x_dead_rec()

        # implement Extended Kalman Filter (EKF)
        self.ekf()

        # store the parameterss for plotting results 
        self.update_past_params()


    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    V. Plot Results
        1. covariance ellipse at the current localization point 
        2. trajectory results:
            1) observation marked by '.' -scatters (True)
            2) true localization values (True)
            3) estimated localization values (True)
            4) dead reackoning boundary (False)
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
        fx = R@(np.array([x, y]))
        px = np.array(fx[0, :] + xEst[0, 0]).flatten()
        py = np.array(fx[1, :] + xEst[1, 0]).flatten()
        return plt.plot(px, py, '--r')

    def plot_trajectory_results(self,color_map,street,it,save_animation,fig_num, save_fig = False,covariance_ellipse = True):
        
        plt.cla()
        # observations
        p1, = plt.plot(self.pst_z[0, :], self.pst_z[1, :], c=color_map[street][0],marker='.') 
        # true positions
        p2, = plt.plot(self.pst_truth[0, :].flatten(),
                 self.pst_truth[1, :].flatten(), c=color_map[street][1]) #'Purple'
        # dead reckoning 
        #plt.plot(hxDR[0, :].flatten(),
                 #hxDR[1, :].flatten(), '-k')
        # estimated values
        p3, = plt.plot(self.pst_est[0, :].flatten(),
                 self.pst_est[1, :].flatten(), c=color_map[street][2]) #'MediumSlateBlue'
        p4 = plt.scatter(self.pst_truth[0, 0],self.pst_truth[1, 0],marker='s')
        p5 = plt.scatter(self.pst_truth[0, -1],self.pst_truth[1, -1],marker='o')
        # whether to plot the covariance ellipse at the current position 
        if covariance_ellipse == True:
            p6, = self.plot_covariance_ellipse(self.x_est, self.cov_est)
        plt.legend([p1,p2,p3,p4,p5,p6], ['Observations', 'True', 'Estimate','Start','End','Covariance Ellipse'])
        plt.axis('equal')
        plt.grid(True)
        plt.title(str(street)+' street: IMU Localization via EKF')
        
        if save_animation and save_fig:
            main_folder = "{}/{}".format(plot_path, street)
            if not os.path.exists(main_folder):
                os.mkdir(main_folder)
            plt.savefig("{}/{}.jpg".format(main_folder, fig_num))
        
        '''
        if save_animation and not it%5:
            if p6:
                ims.append([p1, p2, p3, p4, p5, p6])
            else:
                ims.append([p1, p2, p3, p4, p5])
        '''
        plt.pause(0.001)
        




def main():

    EKF = ekf_localization()
    #print(EKF.get_params(['pst_z','pst_Truth','pst_estimate']))

    # trange can demonstrate the iteration progress
    # iterations according to time stamps
    
    if save_animation:
        ims = []
    fig = plt.figure()
    fig_num = 0

    for i in trange(t.shape[1]):
        # the final index of iteration i 
        limit_len = t.shape[1]
        
        # Ground Truth
        # since tau is difference between two time stamps, let the last tau keep the previous value
        # alternative choice: start with 1 instead of 0
        if i != (t.shape[1]-1):
            tau = t[0][i+1]-t[0][i]
        else:
            tau = t[0][i]-t[0][i-1] 
        
        # get current linear velocity and yaw rate 
        v = v_[i]
        yaw_rate = yaw_rate_[i]
        
        EKF.activate_ekf_localization(v,yaw_rate,tau)
        
        if display_animation:
            save_fig = True if (not i % 50 or i == limit_len - 1) else False
            EKF.plot_trajectory_results(color_map, street, i, save_animation, fig_num, save_fig, covariance_ellipse=True)
            fig_num += 1
    '''
    if save_animation: 
        print(len(ims))
        ani = animation.ArtistAnimation(fig, ims, interval = 100, blit=True,
                                    repeat_delay = 8)

        writer = PillowWriter(fps = 30)
        ani.save("{}/{}.gif".format(ani_path, street), writer = 'imagemagick')
        with open("{0}/{1}_img".format(save_dir, street), "wb") as data:
            pickle.dump((fig, ims), data)
    '''
   
    if save_data == True:
        ekf_param_data = EKF.get_params(save_all=True)
        print("...start saving data...")
        with open("{0}/ekf_localization_data_{1}".format(save_dir, street), 'wb') as ekf_data:
            pickle.dump(ekf_param_data, ekf_data)
        print("...saving data done...")
    
    plt.show()

if __name__ == '__main__':
    main()