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
	image.append(imgg)
	lab.append(label)

print("ok")
images = np.array(image)
labels = np.array(lab).astype(np.float32)
print(labels)
