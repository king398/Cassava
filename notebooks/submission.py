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
	img = tf.keras.preprocessing.image.load_img(path + '/' + filename, target_size=(300, 300))
	img = tf.keras.preprocessing.image.img_to_array(img=img)
	img = img / 255.0

	predictions.append(np.argmax(model.predict(img)))

df = pd.DataFrame(zip(test_file_list, predictions), columns=["image_id", "label"])
df.to_csv("./submission.csv", index=False)
print(df)
