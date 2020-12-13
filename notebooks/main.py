import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.utils import shuffle
import warnings
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import datetime

from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPool2D, Conv2DTranspose, LeakyReLU, \
	BatchNormalization
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.mixed_precision import experimental as mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

model = Sequential()
VGG = tf.keras.applications.VGG19(input_shape=(300, 300, 3), include_top=False, weights=None)
Resnet = tf.keras.applications.ResNet152(input_shape=(300, 300, 3), include_top=False, weights=None, classes=5)
Efficient_net = tf.keras.applications.EfficientNetB3(input_shape=(300, 300, 3), include_top=False)

model.add(layers.experimental.preprocessing.Rescaling(1. / 255))

model.add(Efficient_net)
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(Conv2DTranspose(filters=512, strides=1, kernel_size=2, padding="same"))
model.add(BatchNormalization())

model.add(Conv2DTranspose(filters=512, strides=1, kernel_size=2, padding="same"))
model.add(BatchNormalization())

model.add(Conv2D(filters=512, kernel_size=2, strides=1, activation="relu", padding="same"))
model.add(BatchNormalization())

model.add(Conv2D(filters=512, kernel_size=2, strides=1, activation="relu", padding="same"))
model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=2, strides=1, padding="same"))

model.add(BatchNormalization())
model.add(Conv2DTranspose(filters=256, strides=1, kernel_size=2, padding="same"))
model.add(BatchNormalization())

model.add(Conv2D(filters=256, kernel_size=1, strides=1, activation="relu", padding="same"))
model.add(BatchNormalization())

model.add(Conv2D(filters=256, kernel_size=1, strides=1, activation="relu", padding="same"))
model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=2, strides=1, padding="same"))
model.add(BatchNormalization())

model.add(Conv2DTranspose(filters=128, strides=1, kernel_size=2, padding="same"))
model.add(BatchNormalization())

model.add(Conv2D(filters=128, kernel_size=2, strides=1, activation="relu", padding="same"))
model.add(BatchNormalization())

model.add(Conv2D(filters=128, kernel_size=2, strides=1, activation="relu", padding="same"))
model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=2, strides=1, padding="same"))
model.add(BatchNormalization())
model.add(Conv2D(filters=64, kernel_size=2, strides=1, activation="relu", padding="same"))
model.add(BatchNormalization())

model.add(Conv2D(filters=64, kernel_size=2, strides=1, activation="relu", padding="same"))
model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=2, strides=1, padding="same"))
model.add(BatchNormalization())

model.add(Conv2DTranspose(filters=32, strides=1, kernel_size=2, padding="same"))
model.add(BatchNormalization())

model.add(Conv2D(filters=32, kernel_size=2, strides=1, activation="relu", padding="same"))
model.add(BatchNormalization())

model.add(Conv2D(filters=32, kernel_size=2, strides=1, activation="relu", padding="same"))
model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=2, strides=1, padding="same"))
model.add(BatchNormalization())
model.add(Conv2D(filters=16, kernel_size=2, strides=1, activation="relu", padding="same"))
model.add(BatchNormalization())

model.add(Conv2D(filters=16, kernel_size=2, strides=1, activation="relu", padding="same"))
model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=2, strides=1, padding="same"))
model.add(BatchNormalization())

model.add(Flatten())
model.add(LeakyReLU())
model.add(Dense(512, activation="relu"))
model.add(LeakyReLU())
model.add(tf.keras.layers.Activation('relu'))
model.add(Dense(256, activation="relu"))
model.add(LeakyReLU())
model.add(tf.keras.layers.Activation('relu'))
model.add(LeakyReLU())

model.add(Dense(128, activation="relu"))
model.add(LeakyReLU())

model.add(Dense(64, activation="relu"))

model.add(LeakyReLU())

model.add(Dense(32, activation="relu"))
model.add(LeakyReLU())

model.add(Dense(16, activation="relu"))

model.add(LeakyReLU())

model.add(Dense(8, activation="relu"))

model.add(Dense(5, activation="softmax"))
opt = tf.keras.optimizers.Adagrad(learning_rate=0.01)
model.compile(optimizer=opt,
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])
checkpoint_filepath = "./"
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                               monitor='accuracy',
                                                               mode='max',
                                                               save_best_only=True)

model.fit(images, labels, batch_size=16
          , shuffle=True, epochs=15, callbacks=[model_checkpoint_callback, tensorboard_callback], validation_split=0.15)
model.save(r"/content/models", include_optimizer=True)
