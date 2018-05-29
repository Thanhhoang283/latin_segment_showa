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
		# print(points.index(i))
		if (i != (j-1)):
			# print(points[pos:points.index(i)])
			# print(pos, points.index(i))
			# print(np.mean(points[pos:points.index(i)]))
			results.append(np.mean(points[pos:points.index(j)], dtype=int))
			# results.append(points[pos])
			pos = points.index(i)
		elif (points.index(j) == len(points)-1):
			# print(points.index(i))
			results.append(np.mean(points[pos:points.index(j)], dtype=int))
			# results.append(points[pos])
	return results


def basic_parameters_estimation(img, path, nth):
	# blur_img = filters.gaussian_filter(img, sigma=1, truncate=1)
	# otsu_threshold = threshold_otsu(blur_img)
	# bin_img = (blur_img < otsu_threshold) > 0
	# gray_img = (bin_img*255).astype('uint8')

	h, w = np.shape(img)[:2]
	print(w, h)
	# bin_img = ~(img > 0)*1
	otsu_threshold = threshold_otsu(img)
	bin_img = (img < otsu_threshold) > 0
	gray_img = (bin_img*255).astype('uint8')
	# num, labels = cv2.connectedComponents(gray_img, connectivity = 8)

	ret, labels = cv2.connectedComponents(gray_img, connectivity = 8)

	# Map component labels to hue val
	label_hue = np.uint8(179*labels/np.max(labels))
	blank_ch = 255*np.ones_like(label_hue)
	labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

	# cvt to BGR for display
	labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

	# set bg label to black
	labeled_img[label_hue==0] = 0
	# cv2.imwrite(os.path.join(os.getcwd(), "{}.png".format(nth)), labeled_img)
	# plt.imshow(labeled_img)
	# plt.show()

	count = 0
	for i in range(1,ret):
		cc_img = np.zeros_like(img)
		cc_img[labels == i] = 255
		# plt.imshow(cc_img)
		# plt.show()
		x, y, width, height = cv2.boundingRect(cc_img)
		# cv2.imwrite(os.path.join(path, "{}.png".format(count)), cc_img)
		cv2.imwrite(os.path.join(path, "{}_{}.png".format(nth, count)), img[y:y+height, x:x+width])
		# plt.imshow(img[y:y+width, x:x+width])
		# plt.show()
		count += 1

	# plt.imshow(gray_img)
	# plt.show()


if __name__ == "__main__":

    import os
    import glob
    import argparse
    import shutil

    parser = argparse.ArgumentParser(description="Word Segmentation")
    parser.add_argument('--folder', dest='folder', type=str, help="folder to save")
    parser.add_argument('--path', dest='path', type=str, help="path to linecuts")
    # parser.add_argument('--folder', dest='folder', type=str, help="folder to save")
    args = parser.parse_args()

    # save_folder = os.path.join(os.getcwd(), "training_data_")
    save_folder = args.folder
    # if os.path.exists(save_folder):
    # 	shutil.rmtree(save_folder)
    # os.makedirs(save_folder)

    # path = "/home/thanh/Desktop/test_data_difficult-4/line_1.png"
    # img = cv2.imread(path, 0)
    # plt.imshow(img)
    # plt.show()
    # blur_img = filters.gaussian_filter(img, sigma=1, truncate=1)
    # otsu_threshold = threshold_otsu(blur_img)
    # bin_img = (blur_img < otsu_threshold) > 0
    # gray_img = (bin_img*255).astype('uint8')

    # plt.imshow(gray_img)
    # plt.show()
    # cv2.imwrite("debug.png", gray_img)
    # basic_parameters_estimation(img, save_folder, 0)

    for nth, folder in enumerate(os.listdir(args.path)):
    	# img = cv2.imread(os.path.join(args.path,file), 0)
    	# basic_parameters_estimation(img, save_folder, nth)
    	# name = os.path.basename(os.path.dirname(os.path.join(args.path, folder)))
    	name = os.path.basename(os.path.join(args.path, folder))
    	print(name)
    	for file in os.listdir(os.path.join(args.path, folder)):
    		filename = os.path.join(args.path, folder, file)
    		# print(filename)
    		print(os.path.join(save_folder, name+"_"+file))
    		im = cv2.imread(filename)
    		cv2.imwrite(os.path.join(save_folder, name+"_"+file), im)







