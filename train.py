import cv2
import csv
import numpy as np

lines = []
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    center_filename = line[0].split('/')[-1]
    left_filename = line[1].split('/')[-1]
    right_filename = line[2].split('/')[-1]
    
    center_path = '../data/IMG/' + center_filename
    left_path = '../data/IMG/' + left_filename
    right_path = '../data/IMG/' + right_filename

    center_image = cv2.imread(center_path)
    left_image = cv2.imread(left_path)
    right_image = cv2.imread(right_path)

    images.append(center_image)
    images.append(np.fliplr(center_image))
    images.append(left_image)
    images.append(np.fliplr(left_image))
    images.append(right_image)
    images.append(np.fliplr(right_image))

    measurement = float(line[3])
    measurements.append(measurement)
    measurements.append(-measurement)
    measurements.append(measurement+10)
    measurements.append(-(measurement+10))
    measurements.append(measurement-10)
    measurements.append(-(measurement-10))

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
# Data Normalization
model.add(Lambda(lambda x: x/255.0-0.5, input_shape = (160,320,3)))

# Cropping
model.add(Cropping2D(cropping=((70,25),(0,0))))

# Convolution Layers
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dropout(0.5))
model.add(Dense(84))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')
