import os 
import cv2
from skimage.filters import threshold_otsu, threshold_adaptive
from skimage import filters
import numpy as np
from matplotlib import pyplot as plt 
from scipy import signal 
from smooth import smooth 
from savitzky_golay_filter import savitzky_golay
from scipy.signal import argrelextrema
import scipy.fftpack
from skimage.filters import threshold_otsu, threshold_adaptive


def basic_parameters_estimation(img):
	h, w = np.shape(img)[:2]
	print(w, h)
	# bin_img = ~(img > 0)*1
	otsu_threshold = threshold_otsu(img)
	bin_img = (img < otsu_threshold) > 0
	gray_img = (bin_img*255).astype('uint8')
	# cv2.imwrite("gray.png", gray_img)
	plt.imshow(gray_img)
	plt.show()
	num, labels = cv2.connectedComponents(gray_img, connectivity = 8)
	print(num)

	width_cc = []
	height_cc = []
	for i in range(1,num):
		cc_img = np.zeros_like(img)
		cc_img[labels == i] = 255
		plt.imshow(cc_img)
		plt.show()
		x, y, width, height = cv2.boundingRect(cc_img)
		width_cc.append(width)
		height_cc.append(height)

	# width_his = {wid:width_cc.count(wid) for wid in width_cc}
	# # print(width_his)
	# # print("*****************")
	# max_width = max(width_his, key=int)
	# for key in width_his.keys():
	# 	if (key < max_width/float(4)):
	# 		del width_his[key]
	# # print(width_his)
	# average_width_cc = sum([key*value for key, value in width_his.items()])/float(sum(value for value in width_his.values()))
	# print(average_width_cc)

	# height_his = {hei:height_cc.count(hei) for hei in height_cc}
	# # print(height_his)
	# # print("*****************")
	# max_height = max(height_his, key=int)
	# for key in height_his.keys():
	# 	if (key < max_height/float(4)):
	# 		del height_his[key]
	# # print(height_his)
	# average_height_cc = sum([key*value for key, value in height_his.items()])/float(sum(value for value in height_his.values()))
	# print(average_height_cc)

	# window_size = [3*int(average_width_cc), 12*int(average_height_cc)]
	# std = (3*average_width_cc, 1*average_height_cc)
	# truncate = ((window_size[0]-1)/2 - 0.5)/std[0]

	# blurred_img = filters.gaussian_filter(img, sigma=std, truncate=truncate)
	# # plt.imshow(blurred_img)
	# # plt.show()
	# cv2.imwrite("blur.png", ~(blurred_img*255).astype('uint8'))
	# # bin_blurred_img = ~(blurred_img > threshold_otsu(blurred_img))*1
	# # plt.imshow(bin_blurred_img)
	# # plt.show()
	# blurred_projection = [sum(blurred_img[:,i]) for i in range(w)]
	# # blurred_projection = signal.medfilt(blurred_projection, 15)
	# # blurred_projection = signal.savgol_filter(blurred_projection, h, 9)
	# # blurred_projection = savitzky_golay(blurred_projection, h, 9)
	# # blurred_projection = smooth(np.array(blurred_projection), 71, 'bartlett')
	# # bin_projection = [sum(bin_img[:,i]) for i in range(w)]

	# # plt.plot(blurred_projection)
	# # plt.show()

	# local_minis = np.r_[True, blurred_projection[1:] < blurred_projection[:-1]] \
	# 				& np.r_[blurred_projection[:-1] < blurred_projection[1:], True]
	# # print(local_minis)
	# local_minis = np.where(local_minis == True)
	# print(local_minis[0])
	# # local_minis = argrelextrema(np.array(blurred_projection), np.less)
	# # print(local_minis[0])
	# plt.scatter(local_minis[0], np.zeros(len(local_minis[0])))
	# plt.show()

	# # # Map component labels to hue val
	# # label_hue = np.uint8(179*labels/np.max(labels))
	# # blank_ch = 255*np.ones_like(label_hue)
	# # labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

	# # # cvt to BGR for display
	# # labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

	# # # set bg label to black
	# # labeled_img[label_hue==0] = 0

	# # plt.imshow(labeled_img)
	# # plt.show()


if __name__ == "__main__":

    import os
    import glob
    import argparse

    parser = argparse.ArgumentParser(description="Word Segmentation")
    parser.add_argument('files', help="input file(s) to cut", nargs='+')
    args = parser.parse_args()
    filenames = args.files

    for filename in filenames:
    	img = cv2.imread(filename, 0)
    	# plt.imshow(img)
    	# plt.show()
    	basic_parameters_estimation(img)






