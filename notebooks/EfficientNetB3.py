import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import pandas as pd
from tensorflow.keras.layers import Flatten, Dense, LeakyReLU, BatchNormalization, Dropout
import keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import efficientnet.keras as efn
import tensorflow_addons as tfa
from sklearn.model_selection import StratifiedKFold

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.1, horizontal_flip=True)
train_csv = pd.read_csv(r"/content/train.csv")
train_csv["label"] = train_csv["label"].astype(str)

base_model = efn.EfficientNetB3(weights='noisy-student', include_top=True)

base_model.trainable = True

model = tf.keras.Sequential([

	tf.keras.layers.Input((512, 512, 3)),
	tf.keras.layers.BatchNormalization(renorm=True),
	base_model,
	BatchNormalization(),
	tf.keras.layers.LeakyReLU(),
	tf.keras.layers.Flatten(),
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
	tf.keras.layers.Dense(5, activation='softmax', dtype='float32')
])
fold_number = 0

n_splits = 5
oof_accuracy = []
skf = StratifiedKFold(n_splits=n_splits, random_state=42)

first_decay_steps = 1000

lr = (tf.keras.experimental.CosineDecayRestarts(0.04, first_decay_steps))
opt = tf.keras.optimizers.SGD(lr)
model.compile(
	optimizer=opt,
	loss=tf.keras.losses.CategoricalCrossentropy(),
	metrics=['categorical_accuracy'])

checkpoint_filepath = r"/content/temp/"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
	filepath=checkpoint_filepath,
	save_weights_only=True,
	monitor='val_categorical_accuracy',
	mode='max',
	save_best_only=True)
for train_index, val_index in skf.split(train_csv["image_id"], train_csv["label"]):
	train_set = train_csv.loc[train_index]
	val_set = train_csv.loc[val_index]

	history = model.fit(datagen.flow_from_dataframe(dataframe=train_set,
	                                                directory=r"/content/train_images", x_col="image_id",
	                                                y_col="label", target_size=(512, 512), class_mode="categorical",
	                                                batch_size=16,
	                                                subset="training", shuffle=True),
	                    callbacks=[model_checkpoint_callback],
	                    epochs=20, validation_data=datagen.flow_from_dataframe(dataframe=val_set,
	                                                                           directory=r"/content/train_images",
	                                                                           x_col="image_id",
	                                                                           y_col="label", target_size=(512, 512),
	                                                                           class_mode="categorical", batch_size=16,

	                                                                           subset="validation", shuffle=True))
	oof_accuracy.append(max(history.history["val_accuracy"]))
	fold_number += 1
	if fold_number == n_splits:
		print("Training finished!")
	model.load_weights(checkpoint_filepath)
	model.save(r"/content/models/" + fold_number, include_optimizer=True)
