from keras.models import Sequential, Model, load_model
from keras.layers import Cropping2D, Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPool2D
import numpy as np
import cv2
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from math import ceil
import os
import csv

input_model = 'model9.h5'
output_model = "model10.h5"
data_folder = 'red_curve'
output_image = "loss11.png"

samples = []
with open('./data/' + data_folder + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './data/' + data_folder + '/IMG/'+ batch_sample[0].split('\\')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                images.append(cv2.flip(center_image,1))
                angles.append(center_angle)
                angles.append(center_angle*-1)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

try:
    model = load_model(input_model)
except:
    # set up cropping2D layer
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
            input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70,25), (0,0))))
    model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation="elu"))
    model.add(Dropout(0.2))
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(64,3,3,activation="elu"))
    model.add(Convolution2D(70,3,3,activation="relu"))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(100, activation="elu"))
    model.add(Dense(50))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(1))

    #compile model
    model.compile(loss='mse', optimizer='adam')

history_object = model.fit(train_generator, 
            steps_per_epoch=ceil(len(train_samples)/batch_size), 
            validation_data=validation_generator, 
            validation_steps=ceil(len(validation_samples)/batch_size), 
            epochs=5, verbose=1)

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig(output_image)

model.save(output_model)
exit()