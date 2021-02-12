import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from tensorflow.keras.layers import Flatten, Dense, LeakyReLU, BatchNormalization, Dropout, PReLU
from tensorflow.keras.callbacks import ModelCheckpoint
import albumentations as A
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from pylab import rcParams
import os
import math
from vit_keras import vit, utils
import itertools
import sklearn.metrics
from tensorflow.keras.mixed_precision import experimental as mixed_precision

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)
tf.keras.regularizers.l2(l2=0.01)

datagen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True)
train_csv = pd.read_csv(r"F:\Pycharm_projects\Kaggle Cassava\data\train.csv")
train_csv["label"] = train_csv["label"].astype(str)
image_size = 512
base_model = vit.vit_b32(
	image_size=image_size,
	activation="softmax",
	pretrained=True,
	include_top=True,
	pretrained_top=True,
	classes=5
)

train = train_csv.iloc[:int(len(train_csv) * 0.8), :]
test = train_csv.iloc[int(len(train_csv) * 0.8):, :]
print((len(train), len(test)))
base_model.trainable = False

fold_number = 0

n_splits = 5
oof_accuracy = []
batch_size = 17
first_decay_steps = 500
lr = (tf.keras.experimental.CosineDecayRestarts(0.04, first_decay_steps))
opt = tf.keras.optimizers.SGD(lr, momentum=0.9)

model = tf.keras.Sequential([
	tf.keras.layers.experimental.preprocessing.RandomCrop(height=512, width=512),

	tf.keras.layers.Input((512, 512, 3)),
	tf.keras.layers.BatchNormalization(),
	base_model,
	BatchNormalization(),
	tf.keras.layers.LeakyReLU(),
	tf.keras.layers.Flatten(),

	tf.keras.layers.Dense(5, activation='softmax', dtype='float32')
])
model.compile(
	optimizer=opt,
	loss=tf.keras.losses.CategoricalCrossentropy(),
	metrics=['categorical_accuracy'])

checkpoint_filepath = r"/content/temp/"
model_checkpoint_callback = ModelCheckpoint(
	filepath=checkpoint_filepath,
	save_weights_only=True,
	monitor='val_categorical_accuracy',
	mode='max',
	save_best_only=True)


class BaseConfig(object):
	SEED = 101
	TRAIN_DF = r'F:\Pycharm_projects\Kaggle Cassava\data\train.csv'
	TRAIN_IMG_PATH = r"F:/Pycharm_projects/Kaggle Cassava/data/train_images/"


def albu_transforms_train(data_resize):
	return A.Compose([
		A.ToFloat(),
		A.Resize(800, 800),
	], p=1.)


# For Validation
def albu_transforms_valid(data_resize):
	return A.Compose([
		A.ToFloat(),
		A.Resize(data_resize, data_resize),
	], p=1.)


def CutMix(image, label, DIM, PROBABILITY=0.8):
	# input image - is a batch of images of size [n,dim,dim,3] not a single image of [dim,dim,3]
	# output - a batch of images with cutmix applied
	CLASSES = 5

	imgs = [];
	labs = []
	for j in range(len(image)):
		# DO CUTMIX WITH PROBABILITY DEFINED ABOVE
		P = tf.cast(tf.random.uniform([], 0, 1) <= PROBABILITY, tf.int32)

		# CHOOSE RANDOM IMAGE TO CUTMIX WITH
		k = tf.cast(tf.random.uniform([], 0, len(image)), tf.int32)

		# CHOOSE RANDOM LOCATION
		x = tf.cast(tf.random.uniform([], 0, DIM), tf.int32)
		y = tf.cast(tf.random.uniform([], 0, DIM), tf.int32)

		b = tf.random.uniform([], 0, 1)  # this is beta dist with alpha=1.0

		WIDTH = tf.cast(DIM * tf.math.sqrt(1 - b), tf.int32) * P
		ya = tf.math.maximum(0, y - WIDTH // 2)
		yb = tf.math.minimum(DIM, y + WIDTH // 2)
		xa = tf.math.maximum(0, x - WIDTH // 2)
		xb = tf.math.minimum(DIM, x + WIDTH // 2)

		# MAKE CUTMIX IMAGE
		one = image[j, ya:yb, 0:xa, :]
		two = image[k, ya:yb, xa:xb, :]
		three = image[j, ya:yb, xb:DIM, :]
		middle = tf.concat([one, two, three], axis=1)
		img = tf.concat([image[j, 0:ya, :, :], middle, image[j, yb:DIM, :, :]], axis=0)
		imgs.append(img)

		# MAKE CUTMIX LABEL
		a = tf.cast(WIDTH * WIDTH / DIM / DIM, tf.float32)
		labs.append((1 - a) * label[j] + a * label[k])

	# RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR (maybe use Python typing instead?)
	image2 = tf.reshape(tf.stack(imgs), (len(image), DIM, DIM, 3))
	label2 = tf.reshape(tf.stack(labs), (len(image), CLASSES))
	image2 = tf.cast(image2, dtype=tf.float32)
	label2 = tf.cast(label2, tf.float32)
	return image2, label2


def plot_confusion_matrix(cm, class_names):
	"""
	Returns a matplotlib figure containing the plotted confusion matrix.

	Args:
	  cm (array, shape = [n, n]): a confusion matrix of integer classes
	  class_names (array, shape = [n]): String names of the integer classes
	"""
	figure = plt.figure(figsize=(8, 8))
	plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
	plt.title("Confusion matrix")
	plt.colorbar()
	tick_marks = np.arange(len(class_names))
	plt.xticks(tick_marks, class_names, rotation=45)
	plt.yticks(tick_marks, class_names)

	# Compute the labels from the normalized confusion matrix.
	labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

	# Use white text if squares are dark; otherwise black.
	threshold = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		color = "white" if cm[i, j] > threshold else "black"
		plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	return figure


def visulize(path, n_images, is_random=True, figsize=(16, 16)):
	plt.figure(figsize=figsize)

	w = int(n_images ** .5)
	h = math.ceil(n_images / w)

	image_names = os.listdir(path)
	for i in range(n_images):
		image_name = image_names[i]
		if is_random:
			image_name = random.choice(image_names)

		img = cv2.imread(os.path.join(path, image_name))
		plt.subplot(h, w, i + 1)
		plt.imshow(img)
		plt.xticks([])
		plt.yticks([])
	plt.show()


class CassavaGenerator(tf.keras.utils.Sequence):
	def __init__(self, img_path, data, batch_size,
	             dim, shuffle=True, transform=None,
	             use_mixup=False, use_cutmix=False,
	             use_fmix=False, use_mosaicmix=False):
		self.dim = dim
		self.data = data
		self.shuffle = shuffle
		self.img_path = img_path
		self.augment = transform
		self.use_cutmix = use_cutmix
		self.use_mixup = use_mixup
		self.use_fmix = use_fmix
		self.use_mosaicmix = use_mosaicmix
		self.batch_size = batch_size
		self.list_idx = self.data.index.values
		self.label = pd.get_dummies(self.data['label'], columns=['label'])
		self.on_epoch_end()

	def __len__(self):
		return int(np.ceil(float(len(self.data)) / float(self.batch_size)))

	def __getitem__(self, index):
		batch_idx = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
		idx = [self.list_idx[k] for k in batch_idx]

		Data = np.empty((self.batch_size, *self.dim))
		Target = np.empty((self.batch_size, 5), dtype=np.float32)

		for i, k in enumerate(idx):
			# load the image file using cv2
			image = tf.io.read_file(self.img_path + self.data['image_id'][k])
			image = tf.io.decode_image(image)
			image = np.array(image, dtype="float32")

			res = self.augment(image=image)
			image = res['image']

			# assign
			Data[i, :, :, :] = image
			Target[i, :] = self.label.loc[k, :].values

		# cutmix
		if self.use_cutmix:
			Data, Target = CutMix(Data, Target, self.dim[0])

		return Data, Target

	def on_epoch_end(self):

		self.indices = np.arange(len(self.list_idx))
		if self.shuffle:
			np.random.shuffle(self.indices)


# Define the per-epoch callback.


check_gens = CassavaGenerator(BaseConfig.TRAIN_IMG_PATH, train, 16,
                              (800, 800, 3), shuffle=True,
                              transform=albu_transforms_train(800), use_cutmix=True)

steps = 17119 / 1
valid_steps = 4280 / 20
history = model.fit(check_gens,
                    callbacks=[model_checkpoint_callback],
                    epochs=15, validation_data=datagen.flow_from_dataframe(dataframe=test,
                                                                           directory=BaseConfig.TRAIN_IMG_PATH,
                                                                           x_col="image_id",
                                                                           y_col="label", target_size=(800, 600),
                                                                           class_mode="categorical",
                                                                           batch_size=20,

                                                                           shuffle=True),
                    )
oof_accuracy.append(max(history.history["val_categorical_accuracy"]))
fold_number += 1
if fold_number == n_splits:
	print("Training finished!")
model.load_weights(checkpoint_filepath)
model.save(r"/content/models/" + str(fold_number), include_optimizer=False, overwrite=True)
