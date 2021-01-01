import tensorflow as tf
import numpy as np
import os
import pandas as pd
import cv2
from collections import Counter
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.regularizers import l1
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = keras.models.load_model(r"../input/model-for-training/exp3.h5")
datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

model.summary()
path = "../input/cassava-leaf-disease-classification/test_images"
print(tf.__version__)

test_file_list = os.listdir(path)
predictions = []
for filename in test_file_list:
	img = tf.keras.preprocessing.image.load_img(path + "/" + filename, target_size=(512, 512))

	arr = tf.keras.preprocessing.image.img_to_array(img)
	arr = tf.expand_dims(arr / 255., 0)
	model1_predict = np.argmax(model.predict(arr))
	pre = [model1_predict]
	predictions.append(int(max(set(pre), key=pre.count)))

df = pd.DataFrame(zip(test_file_list, predictions), columns=["image_id", "label"])
df.to_csv("./submission.csv", index=False)
print(df)
