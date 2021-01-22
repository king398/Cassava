import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import pandas as pd
from tensorflow.keras.layers import Flatten, Dense, LeakyReLU, BatchNormalization, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import efficientnet.keras as efn
import tensorflow_addons as tfa
from sklearn.model_selection import StratifiedKFold
import numpy as np

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)
tf.keras.regularizers.l2(l2=0.01)


def augment(image):
	image = np.array(image)
	image = tf.image.random_flip_left_right(image)
	image = tf.image.random_flip_up_down(image)
	return image


datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2, preprocessing_function=augment)
train_csv = pd.read_csv(r"/content/train.csv")
train_csv["label"] = train_csv["label"].astype(str)

base_model = efn.EfficientNetB5(weights='noisy-student', input_shape=(512, 512, 3), include_top=True)

base_model.trainable = True

model = tf.keras.Sequential([
	tf.keras.layers.experimental.preprocessing.RandomCrop(height=512, width=512),

	tf.keras.layers.Input((512, 512, 3)),
	tf.keras.layers.BatchNormalization(renorm=True),
	base_model,
	BatchNormalization(),
	tf.keras.layers.LeakyReLU(),
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(1024),
	BatchNormalization(),

	tf.keras.layers.LeakyReLU(),
	tf.keras.layers.Dense(512),
	BatchNormalization(),

	tf.keras.layers.LeakyReLU(),
	tf.keras.layers.Dense(256),
	BatchNormalization(),

	tf.keras.layers.LeakyReLU(),

	tf.keras.layers.Dense(128),
	BatchNormalization(),

	tf.keras.layers.LeakyReLU(),
	BatchNormalization(),

	tf.keras.layers.Dropout(0.4),
	BatchNormalization(),

	tf.keras.layers.Dense(64),

	tf.keras.layers.LeakyReLU(),
	tf.keras.layers.Dense(32),
	BatchNormalization(),

	tf.keras.layers.Dropout(0.4),

	tf.keras.layers.LeakyReLU(),
	tf.keras.layers.Dense(16),

	tf.keras.layers.LeakyReLU(),
	tf.keras.layers.Dense(8),
	tf.keras.layers.LeakyReLU(),
	tf.keras.layers.Dense(5, dtype='float32'),
	tf.keras.layers.Softmax()
])
fold_number = 0

n_splits = 5
oof_accuracy = []
skf = StratifiedKFold(n_splits=n_splits)

first_decay_steps = 500
lr = (tf.keras.experimental.CosineDecayRestarts(0.04, first_decay_steps))

opt = tf.keras.optimizers.SGD(lr)
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

history = model.fit(datagen.flow_from_dataframe(dataframe=train_csv,
                                                directory=r"/content/train_images", x_col="image_id",
                                                y_col="label", target_size=(800, 600), class_mode="categorical",
                                                batch_size=12,
                                                subset="training", shuffle=True),
                    callbacks=[model_checkpoint_callback],
                    epochs=20, validation_data=datagen.flow_from_dataframe(dataframe=train_csv,
                                                                           directory=r"/content/train_images",
                                                                           x_col="image_id",
                                                                           y_col="label", target_size=(800, 600),
                                                                           class_mode="categorical", batch_size=12,

                                                                           subset="validation", shuffle=True))

model.load_weights(checkpoint_filepath)
