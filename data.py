import tensorflow as tf
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def Make_data(image_path, csv_path):
	labels_file = pd.read_csv(csv_path)
	print(labels_file.head())
	temp1labels = {}
	labels_list = []
	list_img = []
	image_total = 0
	brr = 0
	for i in range(21397):
		image_name = labels_file.iloc[i, 0]
		image_label = labels_file.iloc[i, 1]
		brr += 1
		temp_labels = {image_name, image_label}
		temp1labels.update([temp_labels])
	print(temp1labels)
	for i in tqdm(os.listdir(image_path)):
		real = temp1labels.get(i)
		label = real
		labels_list.append(label)
		path = str(os.path.join(image_path, i))
		img = cv2.imread(path)
		img = cv2.resize(img, dsize=(128, 128))

		image_total += 1
		list_img.append(img)
		print(real)

	labels = np.asarray(labels_list).astype(np.float32)
	images = np.asarray(list_img).astype(np.float32)
	train = tf.data.Dataset.from_tensor_slices((images, labels))
	return train


train = Make_data(image_path=r"../input/cassava-leaf-disease-classification/train_images",
                  csv_path=r"../input/cassava-leaf-disease-classification/train.csv")
print(tf.data.experimental.cardinality(train))
