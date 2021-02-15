import tensorflow as tf
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import keras.backend as K

model1 = tf.keras.models.load_model(r"", compile=False)


path = "../input/cassava-leaf-disease-classification/test_images"
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
		arr = tf.expand_dims(arr / 255., 0)

		arr = tf.convert_to_tensor(arr)
		model1_predict = np.argmax(model1.predict_on_batch(arr))
		pre = [model1_predict]

		predictions.append(max(set(pre), key=pre.count))
		K.clear_session()
	del img, arr
df = pd.DataFrame(zip(test_file_list, predictions), columns=["image_id", "label"])
df.to_csv("./submission.csv", index=False)
print(df)
