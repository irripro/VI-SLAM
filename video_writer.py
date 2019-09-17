import cv2
import os
import numpy as np
from PIL import Image, ImageDraw,ImageFont
from tqdm import trange
import imageio

######################### change your own reading path and saving path ##############################################
extension = ".jpg"
sup_path = "plots"
ani_path = "animation"
images = []
kargs = {"duration": 0.5 }

######################### change your own reading path and saving path ##############################################
def animate(dataset, localization = False, form = "gif"):
	pth = "localization" if localization else "slam"
	folder = "{}/{}/{}".format(sup_path, pth, dataset)

	filenames = sorted(os.listdir(folder))
	if form == "gif":
		for i in trange(int(max(filenames).split(".")[0]) + 1):
			img_pth = "{}/{}{}".format(folder, i, extension)
			if os.path.exists(img_pth):
				images.append(imageio.imread(img_pth))
		imageio.mimsave('{}/{}_{}.gif'.format(ani_path, pth, dataset), images, 'GIF', **kargs)
	
	elif form == "video":
		fps = 6 # increase the number to increase the speed 
		fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
		video_writer = cv2.VideoWriter(filename='{}/{}_{}.avi'.format(ani_path, pth, dataset), fourcc=fourcc, fps=fps, frameSize=(1280,960))
		for i in trange(int(max(filenames).split(".")[0]) + 1):
			img_pth = "{}/{}{}".format(folder, i, extension)
			if os.path.exists(img_pth):
				img = cv2.imread(img_pth)
				video_writer.write(img)

		video_writer.release()

animate(27, localization = True)