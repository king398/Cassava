import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
import keras
import numpy as np

model1 = tf.keras.models.load_model(r"../input/models-gcs/88effnetb3moredata", compile=False)


path = "../input/cassava-leaf-disease-classification/test_images"

test_file_list = os.listdir(path)
predictions = []
model1_predict_list = []
for filename in tqdm(test_file_list):
	img = tf.keras.preprocessing.image.load_img(path + "/" + filename, target_size=(512, 512))
	arr = tf.keras.preprocessing.image.img_to_array(img)
	arr = tf.image.random_flip_left_right(arr)
	arr = tf.expand_dims(arr / 255., 0)
	model1_predict = (np.argmax(model1.predict(arr, training=False,)))

	pre = [model1_predict]
	predictions.append(int(max(set(pre), key=pre.count)))

df = pd.DataFrame(zip(test_file_list, predictions), columns=["image_id", "label"])
df.to_csv("./submission.csv", index=False)
print(df)
