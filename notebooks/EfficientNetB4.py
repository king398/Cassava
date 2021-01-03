import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import pandas as pd
from tensorflow.keras.layers import Flatten, Dense, LeakyReLU, BatchNormalization, Dropout
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import tensorflow_addons as tfa

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

4
# import
from keras.regularizers import l1

# instantiate regularizer

datagen = ImageDataGenerator(validation_split=0.2,
                             dtype=tf.float32, horizontal_flip=True)
train_csv = pd.read_csv(r"/content/train.csv")
train_csv["label"] = train_csv["label"].astype(str)
base_model = tf.keras.applications.EfficientNetB4(include_top=False, weights="imagenet")
base_model.trainable = True

model = tf.keras.Sequential([
	tf.keras.layers.Input((512, 512, 3)),
	tf.keras.layers.BatchNormalization(renorm=True),
	base_model,
	BatchNormalization(),
	tf.keras.layers.LeakyReLU(),
	tf.keras.layers.Flatten(),
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
	tf.keras.layers.Dense(5, activation='softmax')
])
radam = tfa.optimizers.RectifiedAdam(0.004)


class GeneralizedCrossEntropy(tf.losses.Loss):
	def __init__(self, eta=0.7):
		'''
		Paper: https://arxiv.org/abs/1805.07836
		'''
		super(GeneralizedCrossEntropy, self).__init__()
		self.eta = eta

	def call(self, y_true, y_pred):
		t_loss = (1 - tf.pow(tf.reduce_sum(y_true * y_pred, axis=-1),
		                     self.eta)) / self.eta
		return tf.reduce_mean(t_loss)


loss = GeneralizedCrossEntropy
model.compile(
	optimizer=tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5),
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
model.fit(datagen.flow_from_dataframe(dataframe=train_csv,
                                      directory=r"/content/train_images", x_col="image_id",
                                      y_col="label", target_size=(512, 512), class_mode="categorical", batch_size=12,
                                      subset="training", shuffle=True), callbacks=[early, model_checkpoint_callback],
          epochs=10, validation_data=datagen.flow_from_dataframe(dataframe=train_csv,
                                                                 directory=r"/content/train_images",
                                                                 x_col="image_id",
                                                                 y_col="label", target_size=(512, 512),
                                                                 class_mode="categorical", batch_size=12,
                                                                 subset="validation", shuffle=True))
model.load_weights(checkpoint_filepath)
