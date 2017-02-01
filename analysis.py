# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a script file.
"""
import os, sys
import random
import numpy as np
import h5py

TRAIN_DIR = "data/train/"
TEST_DIR = "data/test/"

train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)] # use this for full dataset
train_dogs =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]
train_cats =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]

test_images =  [TEST_DIR+i for i in os.listdir(TEST_DIR)]

def create_dirs(dir_list, category):
    os.mkdir('data/symlinks/train/'+category)
    os.mkdir('data/symlinks/validation/'+category)

    random.shuffle(dir_list)

    for i, file in enumerate(dir_list):
        if i <= 9999:
            dst = file.replace("data/train", "data/symlinks/train/"+category)
            src = file
            os.symlink(os.path.join(os.getcwd(), src), os.path.join(os.getcwd(), dst))
        else:
            dst = file.replace("data/train", "data/symlinks/validation/"+category)
            src = file
            os.symlink(os.path.join(os.getcwd(), src), os.path.join(os.getcwd(), dst))

dir_list = ['data/symlinks', 'data/symlinks/train/', 'data/symlinks/validation/']
for d in dir_list:
    if not os.path.exists(d):
        os.mkdir(d)

#create_dirs(train_cats, "cats")
print("cats done")
#create_dirs(train_dogs, "dogs")
print("dogs done")

import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping


# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = 'data/symlinks/train'
validation_data_dir = 'data/symlinks/validation'
nb_train_samples = 1000
nb_validation_samples = 500
nb_epoch = 2


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

## Callback for loss logging per epoch
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
history = LossHistory()

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'],
              verbose=1,
              shuffle=True)

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary',
        follow_links=True)

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary',
        follow_links=True)

test_generator = test_datagen.flow_from_directory(
        'data/',
        classes = ['test'],
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode=None)

model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples,
        callbacks=[history, early_stopping])

model.save_weights('first_try.h5')

predictions = model.predict_generator(
         test_generator,
         val_samples=128)

loss = history.losses
val_loss = history.val_losses

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('VGG-16 Loss Trend')
plt.plot(loss, 'blue', label='Training Loss')
plt.plot(val_loss, 'green', label='Validation Loss')
plt.xticks(range(0,nb_epoch)[0::2])
plt.legend()

print(predictions)

plt.show()
