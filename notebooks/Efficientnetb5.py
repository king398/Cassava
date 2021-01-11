import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import pandas as pd
from tensorflow.keras.layers import Flatten, Dense, LeakyReLU, BatchNormalization, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import regularizers
import sys
sys.path.append('./bitemperedloss-tf')
from tf_bi_tempered_loss import BiTemperedLogisticLoss

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2, horizontal_flip=True, vertical_flip=True)
train_csv = pd.read_csv(r"/content/train.csv")
train_csv["label"] = train_csv["label"].astype(str)

base_model = tf.keras.applications.ResNet50(include_top=False, weights=None)
base_model.trainable = True

radam = tfa.optimizers.RectifiedAdam()
ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)

model = tf.keras.Sequential([
	tf.keras.layers.Input((512, 512, 3)),
	tf.keras.layers.BatchNormalization(renorm=True),
	base_model,
	tf.keras.layers.LeakyReLU(),
	BatchNormalization(trainable=False),
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001)),
	tf.keras.layers.LeakyReLU(),
	BatchNormalization(trainable=False),
	tf.keras.layers.Dense(256, kernel_regularizer=regularizers.l2(0.0001)),
	tf.keras.layers.LeakyReLU(),
	BatchNormalization(trainable=False),

	tf.keras.layers.Dense(128, kernel_regularizer=regularizers.l2(0.0001)),
	tf.keras.layers.LeakyReLU(),
	BatchNormalization(trainable=False),
	Dropout(0.5),
	tf.keras.layers.LeakyReLU(),

	BatchNormalization(trainable=False),

	tf.keras.layers.Dense(64, kernel_regularizer=regularizers.l2(0.0001)),

	tf.keras.layers.LeakyReLU(),
	BatchNormalization(trainable=False),

	tf.keras.layers.Dense(32, kernel_regularizer=regularizers.l2(0.0001)),
	tf.keras.layers.LeakyReLU(),
	Dropout(0.5),

	BatchNormalization(trainable=False),

	tf.keras.layers.LeakyReLU(),
	BatchNormalization(trainable=False),

	tf.keras.layers.Dense(16, kernel_regularizer=regularizers.l2(0.0001)),
	tf.keras.layers.LeakyReLU(),
	BatchNormalization(trainable=False),
	tf.keras.layers.Dense(8),
	tf.keras.layers.LeakyReLU(),
	BatchNormalization(trainable=False),
	tf.keras.layers.Dense(5, activation='softmax')
])
opt = tf.keras.optimizers.SGD(lr=0.03)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=3, min_lr=0.001)
model.compile(
	optimizer=opt,
	loss=tf.keras.losses.CategoricalCrossentropy(),
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
                                                y_col="label", target_size=(512, 512), batch_size=8,
                                                subset="training", shuffle=True),
                    callbacks=[early, model_checkpoint_callback, reduce_lr],
                    epochs=10, validation_data=datagen.flow_from_dataframe(dataframe=train_csv,
                                                                           directory=r"/content/train_images",
                                                                           x_col="image_id",
                                                                           y_col="label", target_size=(512, 512),
                                                                           class_mode="categorical", batch_size=8,
                                                                           subset="validation", shuffle=True))
model.load_weights(checkpoint_filepath)
