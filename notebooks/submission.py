import tensorflow as tf
import numpy as np
import os
import pandas as pd
import cv2
from collections import Counter

model = tf.keras.models.load_model(r"../input/model-for-training/96.30acctotal.h5")
model2 = tf.keras.models.load_model(r"../input/model-for-training/best.h5")
model.summary()
path = "../input/cassava-leaf-disease-classification/test_images"
test_file_list = os.listdir(path)
predictions = []
for filename in test_file_list:
	img = cv2.imread(
		path + "/" + filename
	)
	imgg = cv2.resize(img, dsize=(300, 300))
	arr = tf.keras.preprocessing.image.img_to_array(imgg)
	arr = tf.expand_dims(arr / 255., 0)
	model1_predict = np.argmax(model.predict(arr))
	model2_predict = np.argmax(model2.predict(arr))
	pre = [model1_predict, model2_predict]
	predictions.append(int(max(set(pre), key=pre.count)))

df = pd.DataFrame(zip(test_file_list, predictions), columns=["image_id", "label"])
df.to_csv("./submission.csv", index=False)
print(df)
model = tf.keras.models.load_model(r"/content/save_raw_model")