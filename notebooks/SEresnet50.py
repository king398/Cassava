import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import pandas as pd
from tensorflow.keras.layers import Flatten, Dense, LeakyReLU, BatchNormalization, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
import efficientnet.keras as efn
import tensorflow_addons as tfa
from sklearn.model_selection import KFold
from collections import Counter
from tf2cv.model_provider import get_model as tf2cv_get_model

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)
tf.keras.regularizers.l2(l2=0.01)

datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2, horizontal_flip=True)
train_csv = pd.read_csv(r"/content/train.csv")
train_csv["label"] = train_csv["label"].astype(str)

base_model = tf2cv_get_model("resnext101_32x4d", pretrained=False, data_format="channels_last")

base_model.trainable = True

model = tf.keras.Sequential([
	tf.keras.layers.experimental.preprocessing.RandomCrop(height=512, width=512),
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
skf = KFold(n_splits=n_splits)

first_decay_steps = 500
lr = (tf.keras.experimental.CosineDecayRestarts(0.03, first_decay_steps))

radam = tfa.optimizers.RectifiedAdam(learning_rate=lr)
ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
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

for train_index, val_index in skf.split(train_csv["image_id"], train_csv["label"]):
	train_set = train_csv.loc[train_index]
	val_set = train_csv.loc[val_index]
	train = datagen.flow_from_dataframe(dataframe=train_set,
	                                    directory=r"/content/train_images", x_col="image_id",
	                                    y_col="label", target_size=(512, 512), class_mode="categorical",
	                                    batch_size=16,
	                                    subset="training", shuffle=True)
	counts = Counter(train.classes)
	print(counts)

	history = model.fit(datagen.flow_from_dataframe(dataframe=train_set,
	                                                directory=r"/content/train_images", x_col="image_id",
	                                                y_col="label", target_size=(512, 512), class_mode="categorical",
	                                                batch_size=12,
	                                                subset="training", shuffle=True),
	                    callbacks=[model_checkpoint_callback],
	                    epochs=5, validation_data=datagen.flow_from_dataframe(dataframe=val_set,
	                                                                          directory=r"/content/train_images",
	                                                                          x_col="image_id",
	                                                                          y_col="label", target_size=(512, 512),
	                                                                          class_mode="categorical", batch_size=12,

	                                                                          subset="validation", shuffle=True))
	oof_accuracy.append(max(history.history["val_categorical_accuracy"]))
	fold_number += 1
	if fold_number == n_splits:
		print("Training finished!")
	model.load_weights(checkpoint_filepath)
	model.save(r"/content/models/" + str(fold_number), include_optimizer=False)
