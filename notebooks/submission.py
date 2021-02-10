import tensorflow as tf
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import gc
import keras.backend as K

model1 = tf.keras.models.load_model(r"../input/models-gcs/8834cutmix", compile=False)
model2 = tf.keras.models.load_model(r"../input/models-gcs/91effnetb3skfcv", compile=False)
model3 = tf.keras.models.load_model(r"../input/models-gcs/8929momentum", compile=False)
model4 = tf.keras.models.load_model(r"../input/models-gcs/8850cutflip", compile=False)
model5 = tf.keras.models.load_model(r"../input/models-gcs/89cutmix", compile=False)
model6 = tf.keras.models.load_model(r"../input/models-gcs/8929momentum", compile=False)
model7 = tf.keras.models.load_model(r"../input/models-gcs/88newlr", compile=False)

path = "../input/cassava-leaf-disease-classification/train_images"
test_file_list = os.listdir(path)

predictions = []
tta_pred = []
model1_predict_list = []
tta = 1
for filename in tqdm(test_file_list):
	tta_pred = []

	for i in range(tta):
		img = tf.keras.preprocessing.image.load_img(path + "/" + filename, target_size=(800, 600))
		arr = np.array(img, dtype=np.float32)
		arr = tf.image.random_flip_left_right(arr)
		arr = tf.expand_dims(arr / 255., 0)
		img2 = tf.keras.preprocessing.image.load_img(path + "/" + filename, target_size=(512, 512))
		arr2 = np.array(img2, dtype=np.float32)
		arr2 = tf.image.random_flip_left_right(arr2)
		arr2 = tf.expand_dims(arr2 / 255., 0)
		arr = tf.convert_to_tensor(arr)
		arr2 = tf.convert_to_tensor(arr2)
		model1_predict = np.argmax(model1.predict_on_batch(arr))
		model2_predict = np.argmax(model2.predict_on_batch(arr2))
		model3_predict = np.argmax(model3.predict_on_batch(arr2))
		model4_predict = np.argmax(model4.predict_on_batch(arr))
		model5_predict = np.argmax(model5.predict_on_batch(arr))
		model6_predict = np.argmax(model6.predict_on_batch(arr))
		model7_predict = np.argmax(model7.predict_on_batch(arr))

		pre = [model1_predict, model2_predict, model3_predict, model4_predict, model5_predict, model6_predict,
		       model7_predict]

		predictions.append(max(set(pre), key=pre.count))
		K.clear_session()
		del img, arr, img2, arr2
df = pd.DataFrame(zip(test_file_list, predictions), columns=["image_id", "label"])
df.to_csv("./submission.csv", index=False)
print(df)
