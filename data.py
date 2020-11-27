import tensorflow as tf
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def make_data(image_path, csv_path):
	"""Make all the images in the given directory

	Args:
		image_path ([type]): [description]
	"""
	for i in tqdm(os.listdir(image_path)):
		path = str(os.path.join(image_path, i))

	labels_file = pd.read_csv(csv_path)
	x = labels_file.iloc[1, 0]
	print(x)


make_data(r"F:\Pycharm_projects\Kaggle Cassava\data\train_images", r"F:\Pycharm_projects\Kaggle Cassava\data\train.csv")
