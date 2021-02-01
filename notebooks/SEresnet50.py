import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import pandas as pd
from tensorflow.keras.layers import Flatten, Dense, LeakyReLU, BatchNormalization, Dropout, PReLU
from tensorflow.keras.callbacks import ModelCheckpoint
import efficientnet.keras as efn
import albumentations as A
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from pylab import rcParams
import os
import math
from tf2cv.model_provider import get_model as tf2cv_get_model

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)
tf.keras.regularizers.l2(l2=0.01)

datagen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True)
train_csv = pd.read_csv(r"/content/train.csv")
train_csv["label"] = train_csv["label"].astype(str)

base_model = tf2cv_get_model("seresnext50_32x4d", pretrained=True, data_format="channels_last")

train = train_csv.iloc[:int(len(train_csv) * 0.8), :]
test = train_csv.iloc[int(len(train_csv) * 0.8):, :]
print((len(train), len(test)))
base_model.trainable = True

fold_number = 0

n_splits = 5
oof_accuracy = []

first_decay_steps = 500
lr = (tf.keras.experimental.CosineDecayRestarts(0.04, first_decay_steps))
opt = tf.keras.optimizers.SGD(lr)

model = tf.keras.Sequential([
	tf.keras.layers.experimental.preprocessing.RandomCrop(height=224, width=224),

	tf.keras.layers.Input((224,224, 3)),
	tf.keras.layers.BatchNormalization(renorm=True),
	base_model,
	BatchNormalization(),
	tf.keras.layers.LeakyReLU(),
	tf.keras.layers.Flatten(),

	tf.keras.layers.Dense(5, activation='softmax', dtype='float32')
])

checkpoint_filepath = r"/content/temp/"
model_checkpoint_callback = ModelCheckpoint(
	filepath=checkpoint_filepath,
	save_weights_only=True,
	monitor='val_categorical_accuracy',
	mode='max',
	save_best_only=True)


class BaseConfig(object):
	SEED = 101
	TRAIN_DF = r"/content/train.csv"
	TRAIN_IMG_PATH = r'/content/train_images/'
	TEST_IMG_PATH = '/content/test_images/'
	CLASS_MAP = '/content/label_num_to_disease_map.json'


def albu_transforms_train(data_resize):
	return A.Compose([
		A.ToFloat(),
		A.Resize(800, 800),
		A.HorizontalFlip()
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

	return image2, label2


def plot_imgs(dataset_show, row, col):
	rcParams['figure.figsize'] = 20, 10
	for i in range(row):
		f, ax = plt.subplots(1, col)
		for p in range(col):
			idx = np.random.randint(0, len(dataset_show))
			img, label = dataset_show[idx]
			ax[p].grid(False)
			ax[p].imshow(img[0])
			ax[p].set_title(idx)
	plt.show()


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
			image = cv2.imread(self.img_path + self.data['image_id'][k])

			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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


# Taylor cross entropy loss
def taylor_cross_entropy_loss(y_pred, y_true, n=3, label_smoothing=0.0):
	"""Taylor Cross Entropy Loss.
	Args:
	y_pred: A multi-dimensional probability tensor with last dimension `num_classes`.
	y_true: A tensor with shape and dtype as y_pred.
	n: An order of taylor expansion.
	label_smoothing: A float in [0, 1] for label smoothing.
	Returns:
	A loss tensor.
	"""
	y_pred = tf.cast(y_pred, tf.float32)
	y_true = tf.cast(y_true, tf.float32)

	if label_smoothing > 0.0:
		num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
		y_true = (1 - num_classes / (num_classes - 1) * label_smoothing) * y_true + label_smoothing / (num_classes - 1)

	y_pred_n_order = tf.math.maximum(tf.stack([1 - y_pred] * n), 1e-7)  # avoide being too small value
	numerator = tf.math.maximum(tf.math.cumprod(y_pred_n_order, axis=0), 1e-7)  # avoide being too small value
	denominator = tf.expand_dims(tf.expand_dims(tf.range(1, n + 1, dtype="float32"), axis=1), axis=1)
	y_pred_taylor = tf.math.maximum(tf.math.reduce_sum(tf.math.divide(numerator, denominator), axis=0),
	                                1e-7)  # avoide being too small value
	loss_values = tf.math.reduce_sum(y_true * y_pred_taylor, axis=1, keepdims=True)
	return tf.math.reduce_sum(loss_values, -1)


class TaylorCrossEntropyLoss(tf.keras.losses.Loss):
	def __init__(self, n=3, label_smoothing=0.0):
		super(TaylorCrossEntropyLoss, self).__init__()
		self.n = n
		self.label_smoothing = label_smoothing

	def call(self, y_true, y_pred):
		return taylor_cross_entropy_loss(y_pred, y_true, n=self.n, label_smoothing=self.label_smoothing)


model.compile(
	optimizer=opt,
	loss=tf.keras.losses.CategoricalCrossentropy(),
	metrics=['categorical_accuracy'])

check_gens = CassavaGenerator(BaseConfig.TRAIN_IMG_PATH, train, 16,
                              (800, 800, 3), shuffle=True,
                              transform=albu_transforms_train(800), use_cutmix=True, use_mixup=False)

plot_imgs(check_gens, row=4, col=3)
history = model.fit(check_gens,
                    callbacks=[model_checkpoint_callback],
                    epochs=25, validation_data=datagen.flow_from_dataframe(dataframe=test,
                                                                           directory=r"/content/train_images",

                                                                           x_col="image_id",
                                                                           y_col="label", target_size=(800, 600),
                                                                           class_mode="categorical", batch_size=16,

                                                                           shuffle=True))
oof_accuracy.append(max(history.history["val_categorical_accuracy"]))
fold_number += 1
if fold_number == n_splits:
	print("Training finished!")
model.load_weights(checkpoint_filepath)
