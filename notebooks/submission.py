import tensorflow as tf
from tqdm import tqdm
import os
import pandas as pd
import numpy as np

model1 = tf.keras.models.load_model(r"../input/models-gcs/88effnetb3moredata", compile=False)
model2 = tf.keras.models.load_model(r"../input/models-gcs/88effnetb3custom", compile=False)
model3 = tf.keras.models.load_model(r"../input/models-gcs/88effnetb3noisyincludetopTrue", compile=False)

path = "../input/cassava-leaf-disease-classification/train_images"
test_file_list = os.listdir(path)
predictions = []
tta_pred = []
model1_predict_list = []
tta = 3
for filename in tqdm(test_file_list):
	tta_pred = []

	for i in range(tta):
		img = tf.keras.preprocessing.image.load_img(path + "/" + filename, target_size=(512, 512))
		arr = np.array(img, dtype=np.float32)
		arr = tf.image.random_flip_left_right(arr)
		arr = tf.expand_dims(arr / 255., 0)
		model1_predict = np.argmax(model1.predict(arr, use_multiprocessing=True))
		model2_predict = np.argmax(model2.predict(arr, use_multiprocessing=True))
		model3_predict = np.argmax(model3.predict(arr, use_multiprocessing=True))
		pre = [model1_predict, model2_predict, model3_predict]
		tta_pred.append(int(max(set(pre), key=pre.count)))

	predictions.append(max(set(tta_pred), key=pre.count))

df = pd.DataFrame(zip(test_file_list, predictions), columns=["image_id", "label"])
df.to_csv("./submission.csv", index=False)
print(df)
