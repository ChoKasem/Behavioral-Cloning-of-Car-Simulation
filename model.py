from keras.models import Sequential, Model, load_model
from keras.layers import Cropping2D, Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPool2D
import numpy as np
import cv2

import csv

#Read CSV data
lines = []
with open('./data/avoid_edge/driving_log.csv') as csvfile:
    next(csvfile) #skipping the first label row
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

#Read images data
images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('\\')[-1] #use \\ for window
    current_path = './data/avoid_edge/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

#Augmented data by flipping image and steering direction
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

try:
    model = load_model('model6.h5')
except:
    #Create Model
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0-0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
    model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
    # model.add(MaxPooling2D())
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
    model.add(Dropout(0.2))
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
    # model.add(Dropout(0.2))
    model.add(Convolution2D(64,3,3,activation="relu"))
    # model.add(Dropout(0.2))
    model.add(Convolution2D(70,3,3,activation="relu"))
    model.add(Dropout(0.1))
    # model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    #compile model
    model.compile(loss='mse', optimizer='adam')

#Train Model
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)

#Save model
model.save('model6.h5')
exit()