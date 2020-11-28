import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.utils import shuffle
import cv2
import warnings

warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPool2D, Conv2DTranspose
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, Activation, Input, BatchNormalization, GlobalAveragePooling2D

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
training_folder = r"data/train_images"
samples_df = pd.read_csv(r"F:\Pycharm_projects\Kaggle Cassava\data\train.csv")
samples_df = shuffle(samples_df, random_state=42)
samples_df["label"] = samples_df["label"].astype("str")
samples_df.head()
temp_labels = {}
image = np.array([]).astype(np.float32)
lab = []
for i in range(len(samples_df)):
	image_name = samples_df.iloc[i, 0]
	image_label = samples_df.iloc[i, 1]
	la = {image_name: image_label}
	temp_labels.update(la)
print(len(temp_labels))
for im in tqdm(os.listdir(training_folder)):
	path = os.path.join(training_folder, im)
	label = temp_labels.get(im)
	img = cv2.imread(path)
	img = tf.image.random_crop(img, size=(100, 100, 3))
	image = np.append(image, img)
	lab.append(label)

lables = np.array(lab).astype(np.float32)

model = Sequential()
model.add(tf.keras.layers.Input(shape=(100, 100, 3)))
layers.experimental.preprocessing.Rescaling(1. / 255)
layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
layers.experimental.preprocessing.RandomRotation(0.2),
model.add(BatchNormalization())

model.add(BatchNormalization())
model.add(Conv2DTranspose(filters=256, strides=1, kernel_size=2))
model.add(BatchNormalization())

model.add(Conv2D(filters=256, kernel_size=1, strides=1, activation="relu"))
model.add(BatchNormalization())

model.add(Conv2D(filters=256, kernel_size=1, strides=1, activation="relu"))
model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=2, strides=1))
model.add(BatchNormalization())

model.add(Conv2DTranspose(filters=128, strides=1, kernel_size=2))
model.add(BatchNormalization())

model.add(Conv2D(filters=128, kernel_size=2, strides=1, activation="relu"))
model.add(BatchNormalization())

model.add(Conv2D(filters=128, kernel_size=2, strides=1, activation="relu"))
model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=2, strides=1))
model.add(BatchNormalization())
model.add(Conv2D(filters=64, kernel_size=2, strides=1, activation="relu"))
model.add(BatchNormalization())

model.add(Conv2D(filters=64, kernel_size=2, strides=1, activation="relu"))
model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=2, strides=1))
model.add(BatchNormalization())

model.add(Conv2DTranspose(filters=32, strides=1, kernel_size=2))
model.add(BatchNormalization())

model.add(Conv2D(filters=32, kernel_size=2, strides=1, activation="relu"))
model.add(BatchNormalization())

model.add(Conv2D(filters=32, kernel_size=2, strides=1, activation="relu"))
model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=2, strides=1))
model.add(BatchNormalization())
model.add(Conv2D(filters=16, kernel_size=2, strides=1, activation="relu"))
model.add(BatchNormalization())

model.add(Conv2D(filters=16, kernel_size=2, strides=1, activation="relu"))
model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=2, strides=1))
model.add(BatchNormalization())

model.add(Flatten())
model.add(BatchNormalization())

model.add(Dense(256, activation="relu"))
model.add(BatchNormalization())

model.add(Dense(128, activation="relu"))
model.add(BatchNormalization())

model.add(Dense(64, activation="relu"))
model.add(BatchNormalization())

model.add(Dense(32, activation="relu"))
model.add(BatchNormalization())

model.add(Dense(16, activation="relu"))
model.add(BatchNormalization())

model.add(Dense(10, activation="softmax"))
tf.keras.optimizers.Adam(
	learning_rate=0.0001, )
model.compile(optimizer='adam',
              loss="sparse_categorical_crossentropy"
              ,
              metrics=['accuracy'])

model.fit(image, lables, batch_size=32, shuffle=True, epochs=1)
