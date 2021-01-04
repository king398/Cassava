import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import pandas as pd
from tensorflow.keras.layers import Flatten, Dense, ReLU, BatchNormalization, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

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
	ReLU(),

	BatchNormalization(),

	Flatten(),
	Dense(256),

	ReLU(),
	BatchNormalization(),

	Dense(128),
	ReLU(),
	BatchNormalization(),

	Dropout(0.3),
	ReLU(),

	BatchNormalization(),

	Dense(64),
	ReLU(),
	BatchNormalization(),
	Dense(32),
	ReLU(),
	BatchNormalization(),
	Dropout(0.3),

	ReLU(),
	BatchNormalization(),

	Dense(16),
	ReLU(),
	BatchNormalization(),
	Dense(8),
	ReLU(),
	BatchNormalization(),

	Dense(5, activation='softmax')
])

loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2)
model.compile(
	optimizer=tf.keras.optimizers.SGD(0.04),
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
