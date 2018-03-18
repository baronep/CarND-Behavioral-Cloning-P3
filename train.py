import cv2
import csv
import numpy as np

images = []
measurements = []
subfolders = ['data2', 'data3', '']

for subfolder in subfolders:
    lines = []
    with open('../data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    for line in lines:
        center_filename = line[0].split('/')[-1]
        left_filename = line[1].split('/')[-1]
        right_filename = line[2].split('/')[-1]
        
        center_path = '../data/' + subfolder + '/IMG/' + center_filename
        left_path = '../data/' + subfolder + '/IMG/' + left_filename
        right_path = '../data/' + subfolder + '/IMG/' + right_filename

        center_image = cv2.imread(center_path)
        left_image = cv2.imread(left_path)
        right_image = cv2.imread(right_path)

        if center_image is None or left_image is None or right_image is None:
               continue

        images.append(center_image)
        images.append(np.fliplr(center_image))
        images.append(left_image)
        images.append(np.fliplr(left_image))
        images.append(right_image)
        images.append(np.fliplr(right_image))

        lr_offset = 0.1
        measurement = float(line[3])
        measurements.append(measurement)
        measurements.append(-measurement)
        measurements.append(measurement+lr_offset)
        measurements.append(-(measurement+lr_offset))
        measurements.append(measurement-lr_offset)
        measurements.append(-(measurement-lr_offset))

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
# Data Normalization
model.add(Lambda(lambda x: (x/255.0)-0.5, input_shape = (160,320,3)))

# Cropping
model.add(Cropping2D(cropping=((70,25),(0,0))))

# Convolution Layers
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('../models/model.h5')
