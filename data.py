import tensorflow as tf
import os
import cv2
import numpy as np
from tqdm import tqdm


def make_data(image_path):
	for i in tqdm(os.listdir(image_path)):
		path = os.path.join(image_path, i)
		print(path)


make_data(r"F:\Pycharm_projects\Kaggle Cassava\data\train_images\''")
