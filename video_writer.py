import cv2
import os
import numpy as np
from PIL import Image, ImageDraw,ImageFont
from tqdm import trange

######################### change your own reading path and saving path ##############################################
#dataset = 27
#extension = '.jpg'
######################### change your own reading path and saving path ##############################################

def video_localization(dataset,extension):
	path = {20:'/Users/momolee/Documents/ECE 276A/hw/ECE276A_HW3/localization/plots_20/', 
	27:'/Users/momolee/Documents/ECE 276A/hw/ECE276A_HW3/localization/plots_27/',
	42:'/Users/momolee/Documents/ECE 276A/hw/ECE276A_HW3/localization/plots_42/'}

	save_path = '/Users/momolee/Documents/ECE 276A/hw/ECE276A_HW3/results/localization/'

	fps = 6 # increase the number to increase the speed 
	fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
	video_writer = cv2.VideoWriter(filename=save_path+str(dataset)+'.avi', fourcc=fourcc, fps=fps, frameSize=(1280,960))

	# read image data
	folder = path[dataset]
	filenames = os.listdir(folder)
	# sort the photo index ensuring the right order over time
	filenames.sort()
	for i in trange(2000):
		if i%10 ==0:
			if os.path.exists(folder + str(i) + extension):
				img = cv2.imread(folder + str(i) + extension)
				video_writer.write(img)

	video_writer.release()

def video_slam(dataset,extension):
	path = {20:'/Users/momolee/Documents/ECE 276A/hw/ECE276A_HW3/slam/mapping_video/plots_20/', 
	27:'/Users/momolee/Documents/ECE 276A/hw/ECE276A_HW3/slam/mapping_video/plots_27/',
	42:'/Users/momolee/Documents/ECE 276A/hw/ECE276A_HW3/slam/mapping_video/plots_42/'}

	save_path = '/Users/momolee/Documents/ECE 276A/hw/ECE276A_HW3/slam/mapping_video/slam'
	fps = 3 # increase the number to increase the speed 
	fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
	video_writer = cv2.VideoWriter(filename=save_path+str(dataset)+'.avi', fourcc=fourcc, fps=fps, frameSize=(1280,960))

	# read image data
	folder = path[dataset]
	filenames = os.listdir(folder)
	# sort the photo index ensuring the right order over time
	filenames.sort()
	for i in trange(2000):
	    if os.path.exists(folder + str(i) + extension):    # only read when the index exists for the img
	        img = cv2.imread(folder + str(i) + extension)
	        video_writer.write(img)

	video_writer.release()

#video_localization(20,'.jpg')
#video_localization(27,'.jpg')
#video_localization(42,'.jpg')
