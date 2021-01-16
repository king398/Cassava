import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import pandas as pd
from tensorflow.keras.layers import Flatten, Dense, LeakyReLU, BatchNormalization, Dropout
import keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import efficientnet.keras as efn
import tensorflow_addons as tfa

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2, horizontal_flip=True)
train_csv = pd.read_csv(r"/content/train.csv")
train_csv["label"] = train_csv["label"].astype(str)


def categorical_focal_loss_with_label_smoothing(gamma=2.0, alpha=0.25, ls=0.1, classes=5.0):
	"""
	Implementation of Focal Loss from the paper in multiclass classification
	Formula:
		loss = -alpha*((1-p)^gamma)*log(p)
		y_ls = (1 - α) * y_hot + α / classes
	Parameters:
		alpha -- the same as wighting factor in balanced cross entropy
		gamma -- focusing parameter for modulating factor (1-p)
		ls    -- label smoothing parameter(alpha)
		classes     -- No. of classes
	Default value:
		gamma -- 2.0 as mentioned in the paper
		alpha -- 0.25 as mentioned in the paper
		ls    -- 0.1
		classes     -- 4
	"""

	def focal_loss(y_true, y_pred):
		# Define epsilon so that the backpropagation will not result in NaN
		# for 0 divisor case
		epsilon = K.epsilon()
		# Add the epsilon to prediction value
		# y_pred = y_pred + epsilon
		# label smoothing
		y_pred_ls = (1 - ls) * y_pred + ls / classes
		# Clip the prediction value
		y_pred_ls = K.clip(y_pred_ls, epsilon, 1.0 - epsilon)
		# Calculate cross entropy
		cross_entropy = -y_true * K.log(y_pred_ls)
		# Calculate weight that consists of  modulating factor and weighting factor
		weight = alpha * y_true * K.pow((1 - y_pred_ls), gamma)
		# Calculate focal loss
		loss = weight * cross_entropy
		# Sum the losses in mini_batch
		loss = K.sum(loss, axis=1)
		return loss

	return focal_loss


base_model = efn.EfficientNetB3(weights='imagenet', input_shape=(512, 512, 3),include_top=False)

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
	tf.keras.layers.Dense(5, activation='softmax')
])
opt = tf.keras.optimizers.SGD(0.03)
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
history = model.fit(datagen.flow_from_dataframe(dataframe=train_csv,
                                                directory=r"/content/train_images", x_col="image_id",
                                                y_col="label", target_size=(512, 512), class_mode="categorical",
                                                batch_size=16,
                                                subset="training", shuffle=True),
                    callbacks=[model_checkpoint_callback],
                    epochs=25, validation_data=datagen.flow_from_dataframe(dataframe=train_csv,
                                                                           directory=r"/content/train_images",
                                                                           x_col="image_id",
                                                                           y_col="label", target_size=(512, 512),
                                                                           class_mode="categorical", batch_size=16,
                                                                           subset="validation", shuffle=True))
model.load_weights(checkpoint_filepath)
import  numpy as np
