import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.mixed_precision  as mixed_precision
import pandas as pd
from tensorflow.keras.layers import Flatten, Dense, LeakyReLU, BatchNormalization, Dropout
from tensorflow.python.keras.utils.data_utils import Sequence

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)
train_csv = pd.read_csv(r"/content/train.csv")
train_csv["label"] = train_csv["label"].astype(str)
train = datagen.flow_from_dataframe(dataframe=train_csv,
                                    directory=r"/content/train_images", x_col="image_id",
                                    y_col="label", target_size=(512, 512), class_mode="categorical", batch_size=16,
                                    subset="training", shuffle=True)
validation = datagen.flow_from_dataframe(dataframe=train_csv,
                                         directory=r"/content/train_images",
                                         x_col="image_id",
                                         y_col="label", target_size=(512, 512), class_mode="categorical", batch_size=16,
                                         subset="validation", shuffle=True)
print(train.image_shape)

base_model = tf.keras.applications.EfficientNetB3(include_top=False)
base_model.trainable = True


def Train_data():
	train = datagen.flow_from_dataframe(dataframe=train_csv,
	                                    directory=r"/content/train_images", x_col="image_id",
	                                    y_col="label", target_size=(512, 512), class_mode="categorical", batch_size=16,
	                                    subset="training", shuffle=True)
	return train


train_ds = tf.data.Dataset.from_generator(lambda :train,
                                          output_types=(tf.float32, tf.float32, tf.float32), output_shapes=(
		tf.TensorShape([2, 512, 512, 3]),
		tf.TensorShape([1, ])
	))
model = tf.keras.Sequential([
	tf.keras.layers.BatchNormalization(axis=-1),
	base_model,
	BatchNormalization(axis=-1),
	tf.keras.layers.LeakyReLU(),
	tf.keras.layers.GlobalAveragePooling2D(),
	tf.keras.layers.Dense(256),
	BatchNormalization(axis=-1),

	tf.keras.layers.LeakyReLU(),

	tf.keras.layers.Dense(128),
	BatchNormalization(axis=-1),

	tf.keras.layers.LeakyReLU(),
	BatchNormalization(axis=-1),

	tf.keras.layers.Dropout(0.4),
	BatchNormalization(axis=-1),

	tf.keras.layers.Dense(64),

	tf.keras.layers.LeakyReLU(),
	tf.keras.layers.Dense(32),
	BatchNormalization(axis=-1),

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
checkpoint_filepath = r"/content/temp/"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
	filepath=checkpoint_filepath,
	save_weights_only=True,
	monitor='val_categorical_accuracy',
	mode='max',
	save_best_only=True)
model.fit(train_ds, callbacks=[early, model_checkpoint_callback], epochs=2, validation_data=validation, batch_size=8,
          )
model.load_weights(checkpoint_filepath)
