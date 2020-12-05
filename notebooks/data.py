import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.utils import shuffle
import warnings
import tensorflow as tf
from PIL import Image
from random import randrange
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPool2D, Conv2DTranspose, LeakyReLU, \
	BatchNormalization
import cv2
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.mixed_precision import experimental as mixed_precision
warnings.filterwarnings("ignore")

samples_df = pd.read_csv(r"/content/train.csv")
samples_df = shuffle(samples_df, random_state=42)
samples_df["label"] = samples_df["label"].astype("str")
samples_df.head()
temp_labels = {}
training_folder = r"/content/train_images"
image = []
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
	imgg = cv2.resize(img, dsize=(300, 300))
	img = tf.keras.preprocessing.image.img_to_array(imgg, dtype=np.float32)

	img = np.asarray(imgg).astype(np.float32)

	image.append(imgg)
	lab.append(label)

print("ok")
images = np.array(image)
labels = np.array(lab).astype(np.float32)
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPool2D, Conv2DTranspose, LeakyReLU



policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)
model = Sequential()
VGG = tf.keras.applications.VGG19(input_shape=(300, 300, 3), include_top=False, weights=None)
Resnet = tf.keras.applications.ResNet152(input_shape=(300, 300, 3), include_top=False, weights=None, classes=5)
Efficient_net = tf.keras.applications.EfficientNetB4(input_shape=(300, 300, 3), include_top=False)

model.add(layers.experimental.preprocessing.Rescaling(1 / 255))

model.add(Resnet)
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(Conv2DTranspose(filters=512, strides=1, kernel_size=2))
model.add(BatchNormalization())

model.add(Conv2DTranspose(filters=512, strides=1, kernel_size=2))
model.add(BatchNormalization())

model.add(Conv2D(filters=512, kernel_size=2, strides=1, activation="relu"))
model.add(BatchNormalization())

model.add(Conv2D(filters=512, kernel_size=2, strides=1, activation="relu"))
model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=2, strides=1))

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
model.add(LeakyReLU())

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
opt = tf.keras.optimizers.Adagrad(
	learning_rate=0.01)
model.compile(optimizer=opt,
              loss="sparse_categorical_crossentropy"
              ,
              metrics=['accuracy'])
checkpoint_filepath = "./"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                               monitor='accuracy',
                                                               mode='max',
                                                               save_best_only=True)
model.fit(images, labels, batch_size=20
          , shuffle=True, epochs=20, callbacks=model_checkpoint_callback)
