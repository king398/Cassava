import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.mixed_precision  as mixed_precision
import pandas as pd
from tensorflow.keras.layers import Flatten, Dense, LeakyReLU, BatchNormalization, Dropout

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)
train_csv = pd.read_csv(r"F:\Pycharm_projects\Kaggle Cassava\data\train.csv")
train_csv["label"] = train_csv["label"].astype(str)
train = datagen.flow_from_dataframe(dataframe=train_csv,
                                    directory=r"F:\Pycharm_projects\Kaggle Cassava\data\train_images", x_col="image_id",
                                    y_col="label", target_size=(400, 400), class_mode="categorical", batch_size=8,
                                    subset="training")
validation = datagen.flow_from_dataframe(dataframe=train_csv,
                                         directory=r"F:\Pycharm_projects\Kaggle Cassava\data\train_images",
                                         x_col="image_id",
                                         y_col="label", target_size=(400, 400), class_mode="categorical", batch_size=8,
                                         subset="validation")

base_model = tf.keras.applications.EfficientNetB3(include_top=False)
base_model.trainable = True

model = tf.keras.Sequential([
	tf.keras.layers.BatchNormalization(),
	base_model,
	BatchNormalization(),
	tf.keras.layers.LeakyReLU(),
	tf.keras.layers.GlobalAveragePooling2D(),
	tf.keras.layers.Dense(256),
	BatchNormalization(),

	tf.keras.layers.LeakyReLU(),
	BatchNormalization(),

	tf.keras.layers.Dense(128),
	BatchNormalization(),
	tf.keras.layers.LeakyReLU(),
	BatchNormalization(),

	tf.keras.layers.Dropout(0.4),
	BatchNormalization(),

	tf.keras.layers.Dense(64),
	BatchNormalization(),

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

model.compile(
	optimizer=tf.keras.optimizers.SGD(lr=0.03),
	loss='categorical_crossentropy',
	metrics=['categorical_accuracy'])
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

early = EarlyStopping(monitor='val_loss',
                      mode='min',
                      patience=5)
checkpoint_filepath = r"F:/Pycharm_projects/Kaggle Cassava/Temp/"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
	filepath=checkpoint_filepath,
	save_weights_only=True,
	monitor='val_categorical_accuracy',
	mode='max',
	save_best_only=True)
model.fit(train, callbacks=[early, model_checkpoint_callback], epochs=10, validation_data=validation, batch_size=8,
          shuffle=True, steps_per_epoch=2139.75,validation_steps=534.875)
model.load_weights(checkpoint_filepath)
