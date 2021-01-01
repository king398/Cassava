import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
import keras


model = keras.models.load_model(r"../input/models-gcs/85effnetb6.h5")
model1 = keras.models.load_model(r"../input/models-gcs/86effnetb6.h5")
model2 = keras.models.load_model(r"../input/models-gcs/83effnetb5no.h5")
model.summary()
path = "../input/cassava-leaf-disease-classification/test_images"

test_file_list = os.listdir(path)
predictions = []
for filename in tqdm(test_file_list):
	img = tf.keras.preprocessing.image.load_img(path + "/" + filename, target_size=(512, 512))
	arr = tf.keras.preprocessing.image.img_to_array(img)
	arr = tf.image.flip_left_right(arr)
	arr = tf.expand_dims(arr / 255., 0)
	model_predict = np.argmax(model.predict(arr))
	model1_predict = np.argmax(model1.predict(arr))
	model2_predict = np.argmax(model.predict(arr))
	pre = [model1_predict, model_predict, model2_predict]
	predictions.append(int(max(set(pre), key=pre.count)))

df = pd.DataFrame(zip(test_file_list, predictions), columns=["image_id", "label"])
df.to_csv("./submission.csv", index=False)
print(df)
