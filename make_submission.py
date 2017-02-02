import os, sys
import random
import numpy as np
import h5py

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

# dimensions of our images.
img_width, img_height = 150, 150

TEST_DIR = "data/test/"
test_images =  [TEST_DIR+i for i in os.listdir(TEST_DIR)]

model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(img_width, img_height, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# Load the model
model.load_weights('first_try.h5')

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(
        samplewise_center=True,
        rescale=1./255)

test_generator = test_datagen.flow_from_directory(
        'data/',
        classes = ['test'],
        target_size=(img_width, img_height),
        batch_size=50,
        shuffle=False,
        class_mode=None)

predictions = model.predict_generator(
         test_generator,
         val_samples=test_generator.nb_sample)

#print(test_generator.filenames)
#print(predictions)

import pandas as pd

subm = pd.read_csv("sample_submission.csv")

ids = [int(x.split("/")[1].split(".")[0]) for x in test_generator.filenames]

for i in range(len(ids)):
    subm.loc[subm.id == ids[i], "label"] = predictions[i]

subm.to_csv("submission2.csv", index=False)
subm.head()
