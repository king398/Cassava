import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import pandas as pd
from tensorflow.keras.layers import Flatten, Dense, LeakyReLU, BatchNormalization, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import efficientnet.tfkeras as efn

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)


def augment(image):
	image = np.array(image)
	image = tf.image.random_brightness(image, 0.3)
	return image


datagen = ImageDataGenerator(validation_split=0.2,
                             horizontal_flip=True, preprocessing_function=augment)
train_csv = pd.read_csv(r"/content/train.csv")
train_csv["label"] = train_csv["label"].astype(str)
base_model = efn.EfficientNetB6(include_top=False, weights="noisy-student")

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
# callbacks
checkpoint_filepath = r"/content/temp/"

early = EarlyStopping(monitor='val_loss',
                      mode='min',
                      patience=5)
model_checkpoint_callback = ModelCheckpoint(
	filepath=checkpoint_filepath,
	save_weights_only=True,
	monitor='val_categorical_accuracy',
	mode='max',
	save_best_only=True)

# lr

# optimizer
radam = tfa.optimizers.RectifiedAdam()
ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)

# loss
loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2)

# model compile
model.compile(
	optimizer=ranger,
	loss=loss,
	metrics=['categorical_accuracy'])

history = model.fit(datagen.flow_from_dataframe(dataframe=train_csv,
                                                directory=r"/content/train_images", x_col="image_id",
                                                y_col="label", target_size=(512, 512), class_mode="categorical",
                                                batch_size=8,
                                                subset="training", shuffle=True),
                    callbacks=[early, model_checkpoint_callback],
                    epochs=10, validation_data=datagen.flow_from_dataframe(dataframe=train_csv,
                                                                           directory=r"/content/train_images",
                                                                           x_col="image_id",
                                                                           y_col="label", target_size=(512, 512),
                                                                           class_mode="categorical", batch_size=8,
                                                                           subset="validation", shuffle=True))
model.load_weights(checkpoint_filepath)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
