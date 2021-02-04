import os
import warnings

import cv2
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from tqdm import tqdm

warnings.filterwarnings("ignore")

samples_df = pd.read_csv(r"F:\Pycharm_projects\Kaggle Cassava\data\train.csv")
samples_df = shuffle(samples_df, random_state=42)
samples_df["label"] = samples_df["label"].astype("str")
samples_df.head()
temp_labels = {}
training_folder = r"F:\Pycharm_projects\Kaggle Cassava\data\train_images"
image = []
lab = []
save4 = r"F:\Pycharm_projects\Kaggle Cassava\data\data\4"

save3 = r"F:\Pycharm_projects\Kaggle Cassava\data\data\3"
save2 = r"F:\Pycharm_projects\Kaggle Cassava\data\data\2"
save1 = r"F:\Pycharm_projects\Kaggle Cassava\data\data\1"
save0 = r"F:\Pycharm_projects\Kaggle Cassava\data\data\0"

for i in range(len(samples_df)):
	image_name = samples_df.iloc[i, 0]
	image_label = samples_df.iloc[i, 1]
	la = {image_name: image_label}
	temp_labels.update(la)
print(len(temp_labels))
for im in tqdm(os.listdir(training_folder)):
	path = os.path.join(training_folder, im)
	label = int(temp_labels.get(im))
	img = cv2.imread(path)
	if label == 0:


		img = cv2.imwrite(save0+"/"+im, img=img)
	if label == 1:


		img = cv2.imwrite(save1+"/"+im, img=img)
	if label == 2:


		img = cv2.imwrite(save2+"/"+im, img=img)
	if label == 3:


		img = cv2.imwrite(save3+"/"+im, img=img)
	if label == 4:


		img = cv2.imwrite(save4+"/"+im, img=img)

