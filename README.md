# VI-SLAM
This report presents the panorama of the design concerning visual-inertial simultaneous localization and mapping(VI-SLAM). Utilizing synchronized measurements from an IMU, a stereo camera, the intrinsic camera calibration and the extrinsic calibration between the two sensors specifying the transformation from the IMU to the left camera frame, this design aims to realize the goals including IMU localization via EKF prediction, landmark mapping via EKF update and ultimately the completion of Visual-Inertial SLAM. 

## Content Overview
* [Prerequisites](#Prerequisites)
* [Folder Organization](#Folder-Organization)
* [Code Organization](#Code-Organization)
* [Dataset](#Dataset)
  * [Training Sets 27 and 42](#Training-Sets-27-and-42)
  * [Testing Set 20](#Testing-Set-20)
  * [Data Extracted](#Data-Extracted)
* [Training Procedure](#Training-Procedure)
* [Files Genearated](#Files-Genearated)
* [Training Performance](#Training-Performance)
  * [EKF-Localization](#EKF-Localization)
  * [EKF-SLAM](#EKF-SLAM)

## Prerequisites
- cv2
- matplotlib
- mpl_toolkits.mplot3d
- pylab
- numpy
- skimage
- imageio
- PIL
- pickle
- tqdm
- scipy
- video writer
- transforms3d

## Folder Organization
<pre>

1. <b>animation</b>
Animation in form of gif and avi for EKF-localization and EKF-SLAM results

2. <b>data</b>
Training and testing data sets

3. <b>save_data</b>
Training data in pickle files
</pre>

## Code Organization
<pre>

1. <b>load_data.py</b>
Loading the data and reading visual features, IMU measurements and calibration paramters

2. <b>EKF_localization.py</b>
Class <b><i>ekf_localization</i></b> inside can implement SLAM localization with use of EKF. 
Users can choose whether to show animation, to save the plots at different time stamps, to save the data 
concerning trajectory and map, to transform the plots into video and etc. Different color map settings 
are allocated to disparate data sets. To get the code run, just simply change the file path 
in which you store your data sets.

3. <b>EKF_slam.py</b>: 
Class <b><i>SLAM_EKF</b></i> inside implement EKF localization and mapping, it combines the functions in 
Class <b><i>ekf_localization</b></i> in <b>EKF_localization.py</b> and adds the mapping procedure with 
prediction and update steps. Notice that the update step is the combined one with i and j for landmark. 
To get the code run, turn to <b>hw3_main.py</b> or add a main function inside this file.

4. <b>hw3_main.py</b>
Implement EKF_SLAM. Users can see the animation of map with trajectory and landmarks on it over time, 
save the plots of different time stamps, transform the plots into video for checking, store the 
experimentation data so that there is no need to run the code again and etc. There are many options 
of the resulting plot, you can turn to <b>EKF_slam. plot_slam_results()</b> to edit or change the setting. 

5. <b>video_writer.py</b>
Transform the plots to video in form of gif/avi. Increase the number of fps to increase the speed of video.

</pre>

## Dataset
  ### Training Set 27 and 42
  - 0027.avi
  - 0027.npz
  - 0042.avi
  - 0042.npz
  
  ### Testing Set 20
  - 0020.avi
  - 0020.npz
  
  ### Data Extracted
  - time stamps
  - pixel coordinates of features
  - linear velocity
  - rotational velocity
  - intrinsic calibration matrix
  - baseline
  - transformation frm IMU to camera frame

## Training Procedure
<pre>

1. Change all file paths to your current file paths and keep the ones for saving data the same as the ones for loading data
2. Run <b>hw3_main.py</b> to implement EKF-SLAM, you may show the animation to see how it works over time
3. Run <b>EKF_localization.py</b> to implement IMU localization
</pre>

## Files Genearated
<pre>

1. <b>EKF-SLAM plots</b>: EKF-SLAM system output (estimated trajectory and landmarks) over time
2. <b>videos</b>: the video in form of '.avi' transformed by output plots
3. data file with extension of <b>.pickle</b>: store attributes you want(You may edit function in the get_params() within the class)
</pre>

## Training Performance
 ### EKF-Localization
 #### Training Set 27
 ![](https://github.com/kwanmolee/VI-SLAM/blob/master/animation/localization_27.gif)
 #### Training Set 42
 ![](https://github.com/kwanmolee/VI-SLAM/blob/master/animation/localization_42.gif)
 #### Testing Set 20
 ![](https://github.com/kwanmolee/VI-SLAM/blob/master/animation/localization_20.gif)
 ### EKF-SLAM
 #### Training Set 27
 ![](https://github.com/kwanmolee/VI-SLAM/blob/master/animation/slam_27.gif)
 #### Training Set 42
 ![](https://github.com/kwanmolee/VI-SLAM/blob/master/animation/slam_42.gif)
 #### Testing Set 20
 ![](https://github.com/kwanmolee/VI-SLAM/blob/master/animation/slam_20.gif)

