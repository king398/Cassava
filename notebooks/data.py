import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.utils import shuffle
import warnings
import cv2


warnings.filterwarnings("ignore")
"""Reads the csv file and creates a labels of images .
"""
samples_df = pd.read_csv(r"/content/train.csv")
samples_df = shuffle(samples_df, random_state=42)
samples_df["label"] = samples_df["label"].astype("str")
samples_df.head()
temp_labels = {}
temp1 = {}
training_folder = r"/content/train_images"
for i in range(len(samples_df)):
	filename = samples_df.iloc[i, 0]
	image_label = samples_df.iloc[i, 1]
	la = {filename: image_label}
	temp_labels.update(la)
print(len(temp_labels))
for im in tqdm(os.listdir(training_folder)):
	pr = os.path.join(training_folder, im)
	labelss = temp_labels.get(im)
	xc = {pr: labelss}
	temp1.update(xc)

filenames = np.array([str(temp1.keys())])
labels = np.array([(str(temp1.values()))
images_list = []
labels_list = []
for i in tqdm(os.listdir(training_folder)):

	filenamee = os.path.join(training_folder, i)
	label = int(temp1.get(filenamee))
	img = cv2.imread(filenamee)
	imgg = cv2.resize(img, dsize=(300, 300))
	imgg = np.asarray(imgg).astype(np.float32)
	imgg = imgg / 255

	images_list.append(imgg)
	labels_list.append(label)
