import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import pandas as pd
from tensorflow.keras.layers import Flatten, Dense, LeakyReLU, BatchNormalization, Dropout, PReLU
from tensorflow.keras.callbacks import ModelCheckpoint
import efficientnet.keras as efn
import tensorflow_addons as tfa
from sklearn.model_selection import StratifiedKFold

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)
tf.keras.regularizers.l2(l2=0.01)

datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2, horizontal_flip=True)
train_csv = pd.read_csv(r"/content/train.csv")
train_csv["label"] = train_csv["label"].astype(str)
base_model = efn.EfficientNetB3(weights='noisy-student', input_shape=(512, 512, 3), include_top=True)

base_model.trainable = True

fold_number = 0

n_splits = 5
oof_accuracy = []
skf = StratifiedKFold(n_splits=n_splits)

first_decay_steps = 500
lr = (tf.keras.experimental.CosineDecayRestarts(0.04, first_decay_steps))
opt = tf.keras.optimizers.SGD(lr)

model = tf.keras.Sequential([
	tf.keras.layers.experimental.preprocessing.RandomCrop(height=512, width=512),

	tf.keras.layers.Input((512, 512, 3)),
	tf.keras.layers.BatchNormalization(renorm=True),
	base_model,
	BatchNormalization(),
	tf.keras.layers.LeakyReLU(),
	tf.keras.layers.Flatten(),

	tf.keras.layers.Dense(5, activation='softmax', dtype='float32')
])
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

tf.keras.backend.clear_session()

history = model.fit(datagen.flow_from_dataframe(dataframe=train_csv,
                                                directory=r"/content/train_images", x_col="image_id",
                                                y_col="label", target_size=(800, 600), class_mode="categorical",
                                                batch_size=16,
                                                subset="training", shuffle=True),
                    callbacks=[model_checkpoint_callback],
                    epochs=15, validation_data=datagen.flow_from_dataframe(dataframe=val_set,
                                                                          directory=r"/content/train_images",
                                                                          x_col="image_id",
                                                                          y_col="label", target_size=(800, 600),
                                                                          class_mode="categorical", batch_size=16,

                                                                          subset="validation", shuffle=True))
oof_accuracy.append(max(history.history["val_categorical_accuracy"]))
fold_number += 1
if fold_number == n_splits:
	print("Training finished!")
model.load_weights(checkpoint_filepath)
model.save(r"/content/models/" + str(fold_number), include_optimizer=False, overwrite=True)
