import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import pandas as pd
from tensorflow.keras.layers import Flatten, Dense, LeakyReLU, BatchNormalization, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import datetime
import os
import numpy as np

tf.keras.regularizers.l2(l2=0.01)
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)



datagen = ImageDataGenerator(validation_split=0.1,
                             dtype=tf.float32, horizontal_flip=True)
train_csv = pd.read_csv(r"/content/train.csv")
train_csv["label"] = train_csv["label"].astype(str)
base_model = tf.keras.applications.ResNet101(include_top=False, weights="imagenet", classes=5)

model = tf.keras.Sequential([
	tf.keras.layers.Input((512, 512, 3)),
	tf.keras.layers.BatchNormalization(renorm=True),
	base_model,
	tf.keras.layers.LeakyReLU(),

	BatchNormalization(),

	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(256),

	tf.keras.layers.LeakyReLU(),
	BatchNormalization(),

	tf.keras.layers.Dense(128),
	tf.keras.layers.LeakyReLU(),
	BatchNormalization(),

	tf.keras.layers.Dropout(0.4),
	tf.keras.layers.LeakyReLU(),

	BatchNormalization(),

	tf.keras.layers.Dense(64),
	tf.keras.layers.LeakyReLU(),
	BatchNormalization(),
	tf.keras.layers.Dense(32),
	tf.keras.layers.LeakyReLU(),
	BatchNormalization(),
	tf.keras.layers.Dropout(0.4),

	tf.keras.layers.LeakyReLU(),
	BatchNormalization(),

	tf.keras.layers.Dense(16),
	tf.keras.layers.LeakyReLU(),
	BatchNormalization(),
	tf.keras.layers.Dense(8),
	tf.keras.layers.LeakyReLU(),
	BatchNormalization(),

	tf.keras.layers.Dense(5, activation='softmax')
])

loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2)
model.compile(
	optimizer=tf.keras.optimizers.SGD(0.03),
	loss='categorical_crossentropy',
	metrics=['categorical_accuracy'])

early = EarlyStopping(monitor='val_loss',
                      mode='min',
                      patience=5)
checkpoint_filepath = r"/content/temp/"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
	filepath=checkpoint_filepath,
	save_weights_only=True,
	monitor='val_categorical_accuracy',
	mode='max',
	save_best_only=True)
history = model.fit(datagen.flow_from_dataframe(dataframe=train_csv,
                                                directory=r"/content/train_images", x_col="image_id",
                                                y_col="label", target_size=(512, 512), class_mode="categorical",
                                                batch_size=32,
                                                subset="training", shuffle=True),
                    callbacks=[early, model_checkpoint_callback],
                    epochs=10, validation_data=datagen.flow_from_dataframe(dataframe=train_csv,
                                                                           directory=r"/content/train_images",
                                                                           x_col="image_id",
                                                                           y_col="label", target_size=(512, 512),
                                                                           class_mode="categorical", batch_size=32,
                                                                           subset="validation", shuffle=True))
model.load_weights(checkpoint_filepath)
