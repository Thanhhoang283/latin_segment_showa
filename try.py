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
# from model import predict 

# def getseperatingPoints (points,up=0,down=1):
#     result = []
#     for i in range(down, len(points)):
#         if ((points[i]-1) != points[i-1]):
#             result.append(np.mean(points[down-1:i], dtype=int))
#             down = i+1
#         elif (i == (len(points)-1)):
#             result.append(np.mean(points[down-1:i+1], dtype=int))
#     return result

def getseperatingPoints(points):
	pos = 0
	results = [pos]
	for i,j in zip(points[:-1], points[1:]):
		if (i != (j-1)):
			results.append(np.mean(points[pos:points.index(j)], dtype=int))
			pos = points.index(i)
		elif (points.index(j) == len(points)-1):
			results.append(np.mean(points[pos:points.index(j)], dtype=int))
	return results

def combine_small_segments(seperatingPoints, minWidth, maxWidth):
    begin = seperatingPoints[0]
    newSeperatingPoints = []
    newSeperatingPoints.append(begin)
    for point in range(1, len(seperatingPoints)):
        distance = abs(begin - seperatingPoints[point])
        if ((distance > maxWidth) or (minWidth < distance < maxWidth)):
            begin = seperatingPoints[point]
            newSeperatingPoints.append(begin)
        else:
            continue 
    return newSeperatingPoints 

def seperate_big_segments(points, averageWidth):
	begin = points[0]
	new_points = []
	# new_points.append(begin)
	for p1, p2 in zip(points[:-1], points[1:]):
		distance = abs(p1 - p2)
		n = int(distance/averageWidth)
		for i in range(n+1):
			new_points.append(int(distance*i/n)+p1)
	return new_points



def basic_parameters_estimation(img_path, path, nth):
	img = cv2.imread(img_path, 0)
	direction, name = os.path.split(img_path)
	# im_dir = os.path.join(path, name[:-4])
	# if os.path.exists(im_dir):
	# 	shutil.rmtree(im_dir)
	# os.makedirs(im_dir)

	h, w = np.shape(img)[:2]
	print(w, h)
	# bin_img = ~(img > 0)*1
	otsu_threshold = threshold_otsu(img)
	bin_img = (img < otsu_threshold) > 0
	gray_img = (bin_img*255).astype('uint8')
	# cv2.imwrite("gray.png", gray_img)
	# plt.imshow(gray_img)
	# plt.show()
	num, labels = cv2.connectedComponents(gray_img, connectivity = 8)
	# print(num)

	width_cc = []
	height_cc = []
	for i in range(1,num):
		cc_img = np.zeros_like(img)
		cc_img[labels == i] = 255
		# plt.imshow(cc_img)
		# plt.show()
		x, y, width, height = cv2.boundingRect(cc_img)
		width_cc.append(width)
		height_cc.append(height)

	# print(width_cc)
	width_his = {wid:width_cc.count(wid) for wid in width_cc}
	max_width = max(width_his, key=int)
	min_width = min(width_his, key=int)
	average_width = sum([key*value for key, value in width_his.items()])/float(sum(value for value in width_his.values()))
	# print(max_width, min_width)
	# print("*****************")
	for key in width_his.keys():
		if (key < max_width/float(4)):
			del width_his[key]
	# print(width_his)
	average_width_cc = sum([key*value for key, value in width_his.items()])/float(sum(value for value in width_his.values()))
	# print("Average width cc: ", average_width_cc)

	height_his = {hei:height_cc.count(hei) for hei in height_cc}
	# print(height_his)
	# print("*****************")
	max_height = max(height_his, key=int)
	for key in height_his.keys():
		if (key < max_height/float(4)):
			del height_his[key]
	# print(height_his)
	average_height_cc = sum([key*value for key, value in height_his.items()])/float(sum(value for value in height_his.values()))
	# print(average_height_cc)

	window_size = [3.5*int(average_width_cc), 12*int(average_height_cc)]
	std = (2*average_width_cc, 0.5*average_height_cc)
	truncate = ((window_size[0]-1)/2 - 0.5)/std[0]
	print(std, truncate)

	# blurred_img = filters.gaussian_filter(255-img, sigma=std, truncate=truncate)
	# blurred_img = filters.gaussian_filter(255-img, sigma=1, truncate=1)
	blurred_img = 255-img
	# cv2.imwrite("blur.png", ~(blurred_img*255).astype('uint8'))
	bin_blurred_img = (blurred_img > threshold_otsu(blurred_img))*1
	gray_img = (bin_blurred_img*255).astype('uint8')

	blurred_projection = [sum(blurred_img[:,i]) for i in range(w)]
	bin_projection = [sum(bin_blurred_img[:,i]) for i in range(w)]
	# print(bin_projection)
	# zeros = [i for i,p in enumerate(bin_projection) if p <= 5]
	zeros = [i for i,p in enumerate(blurred_projection) if p == 0]
	# print(zeros)
	points = getseperatingPoints(zeros)
	points.append(w-1)
	print(points)
	print(len(points))
	tmp = combine_small_segments(points, 10, 25)
	print(len(tmp))
	average = [abs(p1-p2) for p1, p2 in zip(tmp[:-1], tmp[1:])]
	print("***************", average_width)
	average = 25
	abc = [i*average for i in range(int(w/25))]
	print("Average: ", average)
	tmp2 = seperate_big_segments(tmp, 15)
	# tmp2.append(w-1)
	# print("A ", tmp2)
	tmp2 = list(set(tmp2))
	tmp2.sort()
	print("B ", tmp2)

	color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	debug_im = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	seg_im = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	[cv2.line(color_img, (p,0), (p,h), (255,0,0), 2, -1) for p in points]
	[cv2.line(debug_im, (p,0), (p,h), (255,0,0), 2, -1) for p in tmp]
	[cv2.line(seg_im, (p,0), (p,h), (255,0,0), 2, -1) for p in abc]
	cv2.imwrite(os.path.join(path, "line_{}.png".format(nth)), seg_im)
	seg_point = [[p1, p2] for p1, p2 in zip(tmp2[:-1], tmp2[1:])]
	# cv2.imwrite(os.path.join(im_dir, "debug{}.png".format(nth)), img)

	# im_list = []
	# for i, p in enumerate(seg_point):
	# 	# plt.imshow(img[0:h, p[0]:p[1]])
	# 	# plt.show()
	# 	# cv2.imwrite(os.path.join(im_dir, "{}_{}_jp.png".format(nth, i)), img[0:h, p[0]:p[1]])
	# 	color_image = cv2.cvtColor(img[0:h, p[0]:p[1]], cv2.COLOR_GRAY2BGR)
	# 	im_list.append(color_image)

	# boxes = predict(im_list)
	# print("Boxes: {}".format(boxes))
	# debug_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	# for i, value in enumerate(boxes):
	# 	if (value==1):
	# 		cv2.rectangle(debug_img, (seg_point[i][0],0), (seg_point[i][1], h), thickness=2, color=(255,0,0))
	# # plt.imshow(debug_img)
	# # plt.show()
	# cv2.imwrite(os.path.join(path, "debug_{}.png".format(nth)), debug_img)

	f, axarr = plt.subplots(4,1)
	# axarr[0].imshow(gray_img)
	axarr[0].imshow(blurred_img)
	axarr[1].imshow(color_img)
	axarr[2].imshow(debug_im)
	axarr[3].imshow(seg_im)
	plt.show()

if __name__ == "__main__":

    import os
    import glob
    import argparse
    import shutil

    parser = argparse.ArgumentParser(description="Word Segmentation")
    parser.add_argument('--path', dest='path', type=str, help="path to linecuts")
    args = parser.parse_args()

    save_folder = os.path.join(os.getcwd(), "results")
    if os.path.exists(save_folder):
    	shutil.rmtree(save_folder)
    os.makedirs(save_folder)

    for nth, file in enumerate(os.listdir(args.path)):
    	# img = cv2.imread(os.path.join(args.path,file), 0)
    	basic_parameters_estimation(os.path.join(args.path,file), save_folder, nth)






