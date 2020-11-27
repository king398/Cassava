import tensorflow as tf
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def make_data(image_path, csv_path):
	"""Loads the labels from a CSV file and creates a pandas dataframe

	Args:
		image_path ([path]): [paht for the image folder]
		csv_path ([path]): [path for the csv image]
	"""
	labels_file = pd.read_csv(csv_path)
	print(labels_file.head())
	temp1labels = {}
	labels = {}
	for i in range(21397):
		image_name = labels_file.iloc[i, 0]
		image_label = labels_file.iloc[i, 1]
		temp_labels = {image_name, image_label}
		temp1labels.update([temp_labels])
	for i in tqdm(os.listdir(image_path)):
		path = str(os.path.join(image_path, i))
		for ix in temp1labels:
			if ix == i:
				dictt = {path: temp1labels.get(i)}
				labels.update(dictt)
	print(labels)


make_data(r"F:\Pycharm_projects\Kaggle Cassava\data\train_images",
          r"F:\Pycharm_projects\Kaggle Cassava\data\train.csv")
