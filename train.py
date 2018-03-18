import cv2
import csv
import numpy as np
import sklearn
import random

images = []
measurements = []
subfolders = ['data2', 'data3', 'data4', 'data5', 'data6']

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while True:
        samples = random.sample(samples, len(samples))
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                image = cv2.imread(batch_sample[0])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                angle = float(batch_sample[1])
                images.append(image)
                images.append(np.fliplr(image))
                angles.append(angle)
                angles.append(-angle)
                if image is None:
                    print(batch_sample[0])

            X_train = np.array(images)
            y_train = np.array(angles)
            output = sklearn.utils.shuffle(X_train, y_train)
            if output is None:
                print(batch_samples)
            yield output


samples = []

for subfolder in subfolders:
    lines = []
    with open('../data/' + subfolder + '/driving_log.csv') as csvfile:
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

        measurement = float(line[3])

        lr_offset = 0.15
        samples.append( (center_path, measurement) )
        samples.append( (left_path, measurement + lr_offset) )
        samples.append( (right_path, measurement - lr_offset) )


samples_shuffled = random.sample(samples, len(samples))
validation_cut = 0.2
validation_split = int((1-validation_cut)*len(samples_shuffled))
train_samples = samples[:validation_split]
validation_samples = samples[validation_split:]


train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

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

#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)
model.fit_generator(train_generator, samples_per_epoch=2*len(train_samples), validation_data=validation_generator, nb_val_samples=2*len(validation_samples), nb_epoch=3)


model.save('../models/model.h5')
