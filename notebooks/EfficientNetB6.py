import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import pandas as pd
from tensorflow.keras.layers import Flatten, Dense, LeakyReLU, BatchNormalization, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_addons as tfa

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)


def augment(image):
	image = np.array(image).astype(np.float32)
	image = tf.image.random_brightness(image, 0.4)
	return image


datagen = ImageDataGenerator(validation_split=0.2,
                             dtype=tf.float32, horizontal_flip=True, preprocessing_function=augment)
train_csv = pd.read_csv(r"/content/train.csv")
train_csv["label"] = train_csv["label"].astype(str)
base_model = tf.keras.applications.EfficientNetB6(include_top=False, weights="imagenet", classes=5)

model = tf.keras.Sequential([
	Input((512, 512, 3)),
	BatchNormalization(renorm=True, trainable=False),
	base_model,
	Flatten(),
	Dense(256),

	LeakyReLU(),
	BatchNormalization(trainable=False),

	Dense(128),
	LeakyReLU(),
	BatchNormalization(trainable=False),

	LeakyReLU(),

	BatchNormalization(trainable=False),

	Dense(64),
	LeakyReLU(),
	BatchNormalization(trainable=False),
	Dense(32),
	LeakyReLU(),
	BatchNormalization(trainable=False),

	LeakyReLU(),
	BatchNormalization(trainable=False),

	Dense(16),
	LeakyReLU(),
	BatchNormalization(trainable=False),
	Dense(8),
	LeakyReLU(),
	BatchNormalization(trainable=False),

	Dense(5, activation='softmax')
])

lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
	initial_learning_rate=1e-3,
	decay_steps=10000,
	decay_rate=0.9)

radam = tfa.optimizers.RectifiedAdam(lr=lr_scheduler)
ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
loss = tf.keras.layers
model.compile(
	optimizer=ranger,
	loss='categorical_crossentropy',
	metrics=['categorical_accuracy'])

early = EarlyStopping(monitor='val_loss',
                      mode='min',
                      patience=5)
checkpoint_filepath = r"/content/temp/"
model_checkpoint_callback = ModelCheckpoint(
	filepath=checkpoint_filepath,
	save_weights_only=True,
	monitor='val_categorical_accuracy',
	mode='max',
	save_best_only=True)
history = model.fit(datagen.flow_from_dataframe(dataframe=train_csv,
                                                directory=r"/content/train_images", x_col="image_id",
                                                y_col="label", target_size=(512, 512), class_mode="categorical",
                                                batch_size=12,
                                                subset="training", shuffle=True),
                    callbacks=[early, model_checkpoint_callback],
                    epochs=10, validation_data=datagen.flow_from_dataframe(dataframe=train_csv,
                                                                           directory=r"/content/train_images",
                                                                           x_col="image_id",
                                                                           y_col="label", target_size=(512, 512),
                                                                           class_mode="categorical", batch_size=12,
                                                                           subset="validation", shuffle=True))
model.load_weights(checkpoint_filepath)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
