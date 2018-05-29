import os 
import cv2
import copy 
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
from model import predict 

def get_points (points,up=0,down=1):
    result = []
    for i in range(down, len(points)):
        if ((points[i]-1) != points[i-1]):
            result.append(np.mean(points[down-1:i], dtype=int))
            down = i+1
        elif (i == (len(points)-1)):
            result.append(np.mean(points[down-1:i+1], dtype=int))
    return result

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

def seperate_big_segments(points, averageWidth, img):
	points.sort()
	h, w = np.shape(img)[:2]
	otsu_threshold = threshold_otsu(img)
	bin_img = (img < otsu_threshold) > 0
	gray_img = (bin_img*255).astype('uint8')

	new_points = []
	for p1, p2 in zip(points[:-1], points[1:]):
		distance = abs(p1 - p2)
		seg_im = gray_img[0:h, p1:p2]
		x, y, width, height = cv2.boundingRect(seg_im)
		n = int(distance/averageWidth)
		if (n < 2):
			n = 0

		for i in range(n+1):
			try:
				new_points.append(int(distance*i)/n+p1)
			except:
				new_points.append(p1)

	new_points = list(set(new_points))
	new_points.sort()
	return new_points

def get_average_width(img):
	otsu_threshold = threshold_otsu(img)
	bin_img = (img < otsu_threshold) > 0
	gray_img = (bin_img*255).astype('uint8')
	num, labels = cv2.connectedComponents(gray_img, connectivity = 8)
	width_cc = []
	height_cc = []
	for i in range(1,num):
		cc_img = np.zeros_like(img)
		cc_img[labels == i] = 255
		x, y, width, height = cv2.boundingRect(cc_img)
		width_cc.append(width)
		height_cc.append(height)

	width_his = {wid:width_cc.count(wid) for wid in width_cc}
	max_width = max(width_his, key=int)
	# print(width_his)

	for key in width_his.keys():
		if (key < max_width/float(2)):
			del width_his[key]
	average_width = sum([key*value for key, value in 
		width_his.items()])/float(sum(value for value in width_his.values()))
	return average_width

def remove_line(points, img, averageWidth=25):
	h, w = np.shape(img)[:2]
	blurred_img = filters.gaussian_filter(img, sigma=1, truncate=1)
	bin_img = (blurred_img > threshold_otsu(blurred_img))*1
	result = copy.deepcopy(points)

	for p1, p2 in zip(points[:-1], points[1:]):
		dis = abs(p1-p2)
		# print("Distance: ", dis)
		# print(p1,p2)
		if (dis >= 4*averageWidth):
			y_project = [sum(bin_img[0:h, p1:p2][i,:]) for i in range(h)]
			# print(y_project)
			new_seg = img[5:h-5, p1:p2]
			color_im = cv2.cvtColor(new_seg, cv2.COLOR_GRAY2BGR)
			new_points = word_cut(new_seg)
			[cv2.line(color_im, (p,0), (p,h), (255,0,0), 2, -1) for p in new_points]
			pos = points.index(p1)
			[result.insert(pos+i, p+p1) for i,p in enumerate(new_points)]
			result.remove(p1)
			result.remove(p2)

			# f, axarr = plt.subplots(3,1)
			# axarr[0].imshow(new_seg)
			# axarr[1].imshow(color_im)
			# axarr[2].plot(y_project)
			# plt.show()
	result = list(set(result))
	result.sort()
	return result

def word_cut(img):
	h, w = np.shape(img)[:2]
	blurred_img = filters.gaussian_filter(img, sigma=1, truncate=1)
	bin_img = (blurred_img > threshold_otsu(blurred_img))*1

	projection = [sum(bin_img[:,i]) for i in range(w)]
	zeros = [i for i,p in enumerate(projection) if p == 0]
	points = get_points(zeros)
	points.append(1)
	points.append(w-1)
	points.sort()

	return points

def test(points, img):
	im_list = []
	for i, p in enumerate(points):
		color_image = cv2.cvtColor(img[0:h, p[0]:p[1]], cv2.COLOR_GRAY2BGR)
		im_list.append(color_image)

	boxes = predict(im_list)
	print("Boxes: {}".format(boxes))
	debug_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	for i, value in enumerate(boxes):
		if (value==1):
			cv2.rectangle(debug_img, (seg_point[i][0],0), (seg_point[i][1], h), thickness=2, color=(255,0,0))
	plt.imshow(debug_img)
	plt.show()
	# cv2.imwrite(os.path.join(path, "debug_{}.png".format(nth)), debug_img)

def main(img_path, save_path, nth=0):
	img = cv2.imread(img_path, 0)
	h, w = np.shape(img)[:2]

	direction, name = os.path.split(img_path)
	im_dir = os.path.join(save_path, name[:-4])
	if os.path.exists(im_dir):
		shutil.rmtree(im_dir)
	os.makedirs(im_dir)

	average_width = get_average_width(img)
	gray_img = 255 - img
	blurred = filters.gaussian_filter(gray_img, sigma=1, truncate=1)
	binary = (blurred > threshold_otsu(blurred))*1

	points = word_cut(gray_img)
	rm_points = remove_line(points, gray_img)
	sep_points = seperate_big_segments(rm_points, 25, gray_img)

	space_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	rm_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	sep_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	[cv2.line(space_img, (p,0), (p,h), (255,0,0), 2, -1) for p in points]
	[cv2.line(rm_img, (p,0), (p,h), (255,0,0), 2, -1) for p in rm_points]
	[cv2.line(sep_img, (p,0), (p,h), (255,0,0), 2, -1) for p in sep_points]
	# cv2.imwrite(os.path.join(im_dir, "line_{}.png".format(nth)), sep_img)
	# cv2.imwrite(os.path.join(im_dir, "line_{}.png".format(nth)), img)

	cut_points = [[p1, p2] for p1, p2 in zip(sep_points[:-1], sep_points[1:])]
	for i, p in enumerate(cut_points):
		cv2.imwrite(os.path.join(im_dir, "jp_{}_{}.png".format(nth, i)), img[0:h, p[0]:p[1]])
	
	f, axarr = plt.subplots(4,1)
	axarr[0].imshow(binary)
	axarr[1].imshow(space_img)
	axarr[2].imshow(rm_img)
	axarr[3].imshow(sep_img)
	plt.show()

	# test(cut_points, img)


if __name__ == "__main__":

    import os
    import glob
    import argparse
    import shutil

    parser = argparse.ArgumentParser(description="Word Segmentation")
    parser.add_argument('--path', dest='path', type=str, help="path to linecuts")
    args = parser.parse_args()

    save_folder = os.path.join(args.path, "debug")
    if os.path.exists(save_folder):
    	shutil.rmtree(save_folder)
    os.makedirs(save_folder)

    for nth, file in enumerate(os.listdir(args.path)):
    	# img = cv2.imread(os.path.join(args.path,file), 0)
    	print(file)
    	# main(os.path.join(args.path,file), save_folder, nth)
    	try:
    		main(os.path.join(args.path,file), save_folder, nth)
    	except:
    		pass