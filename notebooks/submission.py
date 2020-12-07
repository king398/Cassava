import tensorflow as tf
import numpy as np
import os
import pandas as pd
import cv2

model = tf.keras.models.load_model(r"../input/model-for-training/best.h5")
model.summary()
path = "../input/cassava-leaf-disease-classification/test_images"
test_file_list = os.listdir(path)
predictions = []
for filename in test_file_list:
	img = cv2.imread(path + "/" + filename)

	arr = tf.image.random_crop(img, size=(300, 300, 3))
	arr = tf.keras.preprocessing.image.img_to_array(arr, dtype=np.float32)
	arr = np.asarray(arr).astype(np.float32)
	predictions.append(np.argmax(model.predict(arr)[0]))

df = pd.DataFrame(zip(test_file_list, predictions), columns=["image_id", "label"])
df.to_csv("./submission.csv", index=False)
print(df)
