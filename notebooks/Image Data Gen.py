import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import cv2


class preprocessing():
	def input(pathh):
		images = []
		for i in tqdm(os.scandir(path=pathh)):
			i = os.path.join(pathh, i)
			image = cv2.imread(i)
			image = tf.image.random_crop(image, size=(512, 512, 3))
			image = image / 255
			images.append(image)


preprocessing.input(pathh=r"F:\Pycharm_projects\Kaggle Cassava\data\train_images")
