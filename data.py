import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.utils import class_weight
from sklearn.preprocessing import minmax_scale
import random
import cv2
import warnings

warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Activation, Input, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers.experimental.preprocessing import RandomCrop, CenterCrop, RandomRotation

training_folder = r"F:\Pycharm_projects\Kaggle Cassava\data\train_images"
samples_df = pd.read_csv(r"F:\Pycharm_projects\Kaggle Cassava\data\train.csv")
samples_df = shuffle(samples_df, random_state=42)
samples_df["label"] = samples_df["label"].astype("str")
samples_df.head()
temp_labels = {}
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
	img = cv2.resize(img,d)
	cv2.imshow("img", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	break
