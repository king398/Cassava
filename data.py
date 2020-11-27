import tensorflow as tf
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

tf.compat.v1.disable_eager_execution()


def make_data(image_path, csv_path):
	"""Make data in a CSV file

	Args:
		image_path ([type]): [description]
		csv_path ([type]): [description]
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
		img = cv2.imread(path)
		cv2.imshow(winname="hat", mat=img)
		cv2.waitKey(0)

		# closing all open windows
		cv2.destroyAllWindows()
		for ix in temp1labels:
			if ix == i:
				label = temp1labels.get(i)

				print(label)


make_data(r"F:\Pycharm_projects\Kaggle Cassava\data\train_images",
          r"F:\Pycharm_projects\Kaggle Cassava\data\train.csv")
