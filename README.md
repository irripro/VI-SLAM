# VI-SLAM

**Project Description**
This report presents the panorama of the design concerning visual-inertial simultaneous localization and mapping(VI-SLAM). Utilizing synchronized measurements from an IMU, a stereo camera, the intrinsic camera calibration and the extrinsic calibration between the two sensors specifying the transformation from the IMU to the left camera frame, this design aims to realize the goals including IMU localization via EKF prediction, landmark mapping via EKF update and ultimately the completion of Visual-Inertial SLAM. 

**The project is divided into the following parts based on different functions：**
1.  load_data.py: loading the data and reading visual features, IMU measurements and calibration paramters
2.	EKF_localization.py: Class ekf_localization inside can implement SLAM localization with use of EKF. Users can choose whether to show animation, to save the plots at different time stamps, to save the data concerning trajectory and map, to transform the plots into video and etc. Different color map settings are allocated to disparate data sets. To get the code run, just simply change the file path in which you store your data sets.
3.	EKF_slam.py: Class SLAM_EKF inside implement EKF localization and mapping, it combines the functions in Class ekf_localization in EKF_localization.py and adds the mapping procedure with prediction and update steps. Notice that the update step is the combined one with i in  and j for landmark. To get the code run, turn to hw3_main.py or add a main function inside this file.
4.	hw3_main.py: for the implementation of the EKF_SLAM. Users can see the animation of map with trajectory and landmarks on it over time, save the plots of different time stamps, transform the plots into video for checking, store the experimentation data so that there is no need to run the code again and etc. There are many options of the resulting plot, you can turn to EKF_slam. plot_slam_results() to edit or change the setting. 
5.	video_writer.py: Transform the plots to video in form of avi. Increase the number of fps to increase the speed of video.

**The steps to get the code run:**

- change all file paths to your current file paths and keep the ones for saving data the same as the ones for loading data
- run hw3_main.py to implement EKF-SLAM
  you may show the animation to see how it works over time
- run EKF_localization.py to implement IMU localization

**Files generated during running:**

- EKF-SLAM plots: EKF-SLAM system output (estimated trajectory and landmarks) over time
- videos: the video in form of '.avi' transformed by output plots
- data file with extension of '.pickle' storing attributes you want( by editing function in the get_params() within the class)

