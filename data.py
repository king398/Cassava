import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.utils import shuffle
import cv2
import warnings

warnings.filterwarnings('ignore')

import tensorflow as tf



def make_data(training_folder, labels_file):
	samples_df = pd.read_csv(labels_file)
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

	labels = np.array(lab).astype(np.float32)
	dataset = tf.data.Dataset.from_tensor_slices((image, labels))
	return dataset
