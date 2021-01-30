import tensorflow as tf
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
from tensorflow.keras.mixed_precision import experimental as mixed_precision

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)
model1 = tf.keras.models.load_model(r"F:\Pycharm_projects\Kaggle Cassava\models\models\87effnetb3noisy", compile=False)

path = r"F:\Pycharm_projects\Kaggle Cassava\data\train_images"
test_file_list = os.listdir(path)
predictions = []

for filename in tqdm(test_file_list):
	img = tf.keras.preprocessing.image.load_img(path + "/" + filename, target_size=(512,512))
	arr = tf.keras.preprocessing.image.img_to_array(img)
	arr = tf.image.random_flip_left_right(arr)
	arr = tf.expand_dims(arr / 255., 0)
	model1_predict = np.argmax(model1.predict(arr))

	pre = [model1_predict]
	predictions.append(int(max(set(pre), key=pre.count)))

df = pd.DataFrame(zip(test_file_list, predictions), columns=["image_id", "label"])
df.to_csv("./submission.csv", index=False)
