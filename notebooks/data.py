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
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPool2D, Conv2DTranspose, LeakyReLU
import cv2

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
    img = Image.open(path)
    img = img.crop((300, 300, 300, 300))
    img = tf.keras.preprocessing.image.img_to_array(img, dtype=np.float32)

    img = np.asarray(img).astype(np.float32)

    image.append(img)
    lab.append(label)

print("ok")

images = tf.convert_to_tensor(image, dtype=tf.float16)
labels = tf.keras.utils.to_categorical(np.array(lab).astype(np.float32))

model = Sequential()
VGG = tf.keras.applications.VGG19(input_shape=(
    300, 300, 3), include_top=False, weights=None)
Resnet = tf.keras.applications.ResNet152(input_shape=(
    300, 300, 3), include_top=False, weights=None, classes=5)
Efficient_net = tf.keras.applications.EfficientNetB4(
    input_shape=(300, 300, 3), include_top=False)

model.add(layers.experimental.preprocessing.Rescaling(1 / 255))

model.add(Resnet)
model.add(LeakyReLU())

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
              loss="categorical_crossentropy",
              metrics=['accuracy'])
checkpoint_filepath = "./"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                               monitor='accuracy',
                                                               mode='max',
                                                               save_best_only=True)
model.fit(image, labels, batch_size=4, shuffle=True,
          epochs=10, callbacks=model_checkpoint_callback)
model.predict()
