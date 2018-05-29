import os
from keras.models import Sequential 
from keras.layers import Conv2D 
from keras.layers import MaxPooling2D 
from keras.layers import Flatten
from keras.layers import Dense 
from keras.preprocessing.image import ImageDataGenerator
from IPython.display import display
from PIL import Image
import cv2
import numpy as np
from sklearn.metrics import accuracy_score
from skimage import io, transform
from matplotlib import pyplot as plt

def cnn_classifier():
    cnn = Sequential()
    cnn.add(Conv2D(32, (3,3), input_shape = (50, 50, 3), padding='same', activation = 'relu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    cnn.add(Conv2D(64, (3,3), padding='same', activation = 'relu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    cnn.add(Flatten())
    cnn.add(Dense(512, activation = 'relu'))
    cnn.add(Dense(2, activation = 'softmax'))
    cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    print(cnn.summary())
    return cnn

def reshaped_image(image):
    return transform.resize(image,(50, 50, 3)) # (cols (width), rows (height)) and don't use np.resize()

def load_images_from_folder(path):
    Images = os.listdir(path)
    train_images = []
    train_labels = []
    for image in Images:
            l = [0,0] # [cat,dog]
            # img = cv2.imread(os.path.join(path, image))
            # train_images.append(reshaped_image(img))
            
            if image.find('jp') != -1:
                # path = os.path.join(path, image)
                img = cv2.imread(os.path.join(path, image))
                train_images.append(reshaped_image(img))
                l = [1,0] 
                train_labels.append(l)
            if image.find('latin') != -1:
                # path = os.path.join(path, image)
                img = cv2.imread(os.path.join(path, image))
                train_images.append(reshaped_image(img))
                l = [0,1] 
                train_labels.append(l)
    return np.array(train_images), np.array(train_labels)
        
def train_test_split(train_data, train_labels, fraction):
    index = int(len(train_data)*fraction)
    return train_data[:index], train_labels[:index], train_data[index:], train_labels[index:]


# def main(path):
# 	train_data, train_labels = load_images_from_folder(path)
# 	fraction = 1
# 	train_data, train_labels, test_data, test_labels = train_test_split(train_data, train_labels, fraction)
# 	print ("Train data size: ", len(train_data))
# 	print ("Test data size: ", len(test_data))

# 	cnn = cnn_classifier()

# 	print ("Train data shape: ", train_data.shape)
# 	print ("Test data shape: ", test_data.shape)

# 	# idx = np.random.permutation(train_data.shape[0])
# 	# cnn.fit(train_data[idx], train_labels[idx], batch_size = 64, epochs = 20)
# 	# cnn.save_weights('weights.h5')


# 	cnn.load_weights('./weights.h5')
# 	for im_path in os.listdir(path):
# 		img = reshaped_image(cv2.imread(os.path.join(path, im_path)))
# 		predicted_test_labels = np.argmax(cnn.predict(img[None,:,:,:]), axis=1)
# 		# print(im_path)
# 		if (predicted_test_labels[0] == 0):
# 			print("*********** Predict: Japanese")
# 		else:
# 			print("*********** Predict: Latin")
# 		# print(im_path)

# 		# print(im_path)
# 		# if im_path.find('jp') != -1:
# 		# 	print("Label: Japanese")
# 		# else:
# 		# 	print("Label: Latin")
# 		# plt.imshow(img)
# 		# plt.show()


# 	# predicted_test_labels = np.argmax(cnn.predict(test_data), axis=1)
# 	# test_labels = np.argmax(test_labels, axis=1)

# 	# print ("Actual test labels:", test_labels)
# 	# print ("Predicted test labels:", predicted_test_labels)
# 	# print ("Accuracy score:", accuracy_score(test_labels, predicted_test_labels))

def predict(im_list):
    boxes = []
    probs = []
    cnn = cnn_classifier()
    cnn.load_weights('./weights.h5')
    for im in im_list:
        # plt.imshow(im)
        # plt.show()
        img = reshaped_image(im)
        predicted_test_labels = np.argmax(cnn.predict(img[None,:,:,:]), axis=1)
        prob = cnn.predict_proba(img[None,:,:,:])
        # print("*********", prob[0])
        # boxes.append(predicted_test_labels[0])
        if (prob[0][1] >= 0.9):
            boxes.append(1)
        else:
            boxes.append(0)
        probs.append(prob[0])
        # print("************** ", predicted_test_labels[0])
    # print(result)
    return boxes, probs
	
# if __name__ == '__main__':
# 	import os
# 	import glob
# 	import argparse

	# parser = argparse.ArgumentParser(description="Language classification")
	# parser.add_argument('--path', dest='path', help="input file(s) to process")
	# args = parser.parse_args()
	# main(args.path)