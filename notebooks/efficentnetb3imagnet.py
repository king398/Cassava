import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import mixed_precision
import pandas as pd
from tensorflow.keras.layers import Flatten, Dense, LeakyReLU, BatchNormalization, Dropout, PReLU
from tensorflow.keras.callbacks import ModelCheckpoint
import efficientnet.keras as efn
from sklearn.model_selection import StratifiedKFold
import datetime
import numpy as np
import cv2
import albumentations as A

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

tf.keras.regularizers.l2(l2=0.01)


def albu_transforms_train(data_resize):
	return A.Compose([
		A.ToFloat(),
		A.Resize(data_resize, data_resize),
	], p=1.)


# For Validation
def albu_transforms_valid(data_resize):
	return A.Compose([
		A.ToFloat(),
		A.Resize(data_resize, data_resize),
	], p=1.)


def CutMix(image, label, DIM, PROBABILITY=1.0):
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


plot_imgs(check_gens, row=4, col=3)
datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2, horizontal_flip=True)
train_csv = pd.read_csv(r"F:\Pycharm_projects\Kaggle Cassava\data\train.csv")
train_csv["label"] = train_csv["label"].astype(str)
base_model = efn.EfficientNetB3(weights='noisy-student', input_shape=(512, 512, 3), include_top=True)

base_model.trainable = True

fold_number = 0

n_splits = 5
oof_accuracy = []
skf = StratifiedKFold(n_splits=n_splits)
first_decay_steps = 500
lr = (tf.keras.experimental.CosineDecayRestarts(0.03, first_decay_steps))
opt = tf.keras.optimizers.SGD(lr)


check_gens = CassavaGenerator(r"F:\Pycharm_projects\Kaggle Cassava\data\train_images", train_csv
                              , 20,
                              (128, 128, 3), shuffle=True,
                              use_mixup=False, use_cutmix=True,
                              use_fmix=False, transform=albu_transforms_train(128))
model = tf.keras.Sequential([
	tf.keras.layers.experimental.preprocessing.RandomCrop(height=512, width=512),

	tf.keras.layers.Input((512, 512, 3)),
	tf.keras.layers.BatchNormalization(renorm=True),
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

checkpoint_filepath = r"F:\Pycharm_projects\Kaggle Cassava\temp/"
model_checkpoint_callback = ModelCheckpoint(
	filepath=checkpoint_filepath,
	save_weights_only=True,
	monitor='val_categorical_accuracy',
	mode='max',
	save_best_only=True)

log_dir = "F:\Pycharm_projects\Kaggle Cassava\logs"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
history = model.fit(datagen.flow_from_dataframe(dataframe=train_csv,
                                                directory=r"F:\Pycharm_projects\Kaggle Cassava\data\train_images",
                                                x_col="image_id",
                                                y_col="label", target_size=(800, 600), class_mode="categorical",
                                                batch_size=6,
                                                subset="training", shuffle=True),
                    callbacks=[model_checkpoint_callback, tensorboard_callback],
                    epochs=15, validation_data=datagen.flow_from_dataframe(dataframe=train_csv,
                                                                           directory=r"F:\Pycharm_projects\Kaggle Cassava\data\train_images",
                                                                           x_col="image_id",
                                                                           y_col="label", target_size=(800, 600),
                                                                           class_mode="categorical", batch_size=6,

                                                                           subset="validation", shuffle=True))
oof_accuracy.append(max(history.history["val_categorical_accuracy"]))
fold_number += 1
if fold_number == n_splits:
	print("Training finished!")
model.load_weights(checkpoint_filepath)
model.save(r"/content/models/" + str(fold_number), include_optimizer=False, overwrite=True)
