import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.utils import shuffle
import cv2
import warnings
import tensorflow as tf

warnings.filterwarnings('ignore')


def make_data(training_folder, labels_file):
	samples_df = pd.read_csv(labels_file)
	samples_df = shuffle(samples_df, random_state=42)
	samples_df["label"] = samples_df["label"].astype("str")
	samples_df.head()
	temp_labels = {}

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
		img = tf.image.random_crop(img, size=(250, 250, 3))
		image.append(img)
		lab.append(label)
	labels = np.array(lab).astype(np.float32)
	train = tf.data.Dataset.from_tensor_slices((image, labels))
	return train


data = make_data(training_folder=r'data/train_images', labels_file=r'data/train.csv')
