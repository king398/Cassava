import tensorflow as tf
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import keras.backend as K

model1 = tf.keras.models.load_model(r"../input/models-gcs/8925b5cutmix", compile=False)
print("model1_loaded")
model2 = tf.keras.models.load_model(r"../input/models-gcs/91effnetb3skfcv", compile=False)
print("model2_loaded")

model3 = tf.keras.models.load_model(r"../input/models-gcs/88effnetb3noisyincludetopTrue", compile=False)
print("model3_loaded")

model4 = tf.keras.models.load_model(r"../input/models-gcs/8864b5cutmix", compile=False)
print("model4_loaded")

model5 = tf.keras.models.load_model(r"../input/models-gcs/89cutmix", compile=False)
print("model5_loaded")

model6 = tf.keras.models.load_model(r"../input/models-gcs/8886retraincutmix", compile=False)
print("model6_loaded")

model7 = tf.keras.models.load_model(r"../input/models-gcs/88newlr", compile=False)
print("model7_loaded")

path = "../input/cassava-leaf-disease-classification/test_images"
test_file_list = os.listdir(path)

predictions = []
tta_pred = []
model1_predict_list = []
tta = 5
for filename in tqdm(test_file_list):
	img = tf.keras.preprocessing.image.load_img(path + "/" + filename, target_size=(800, 600))
	arr = np.array(img, dtype=np.float32)
	arr = tf.expand_dims(arr / 255., 0)
	img2 = tf.keras.preprocessing.image.load_img(path + "/" + filename, target_size=(512, 512))
	arr2 = np.array(img2, dtype=np.float32)
	arr2 = tf.expand_dims(arr2 / 255., 0)

	arr = tf.convert_to_tensor(arr)
	arr2 = tf.convert_to_tensor(arr2)
	tta_pred = []

	for i in range(tta):
		model1_predict = model1.predict_on_batch(arr)
		model2_predict = model2.predict_on_batch(arr2)
		model3_predict = model3.predict_on_batch(arr2)
		model4_predict = model4.predict_on_batch(arr)
		model5_predict = model5.predict_on_batch(arr)
		model6_predict = model6.predict_on_batch(arr)
		model7_predict = model7.predict_on_batch(arr)
		tta_pred.append(model1_predict + model2_predict + model3_predict + model4_predict + model5_predict + model6_predict + model7_predict)

	print(tta_pred)
	tta_pred = np.argmax(tta_pred)

	K.clear_session()

predictions.append(tta_pred)

del img, arr, img2, arr2
df = pd.DataFrame(zip(test_file_list, predictions), columns=["image_id", "label"])
df.to_csv("./submission.csv", index=False)
print(df)
