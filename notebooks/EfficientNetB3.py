# Importing all the required libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.layers import Flatten, Dense, LeakyReLU, BatchNormalization, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd

# regularizersto prevent overfitting by by penalizing model for it
tf.keras.regularizers.l2(l2=0.01)

# using mixed precison to improve model accuracy and speedup training process
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

# ImageDataGenerator For Data Augmentation And making our Dataset
datagen = ImageDataGenerator(validation_split=0.1,
                             dtype=tf.float32, horizontal_flip=True)
# reading our labels file using pandas
train_csv = pd.read_csv(r"/content/train.csv")
train_csv["label"] = train_csv["label"].astype(str)

# importing the model which which we are going to use
base_model = tf.keras.applications.EfficientNetB3(include_top=False, weights="imagenet", classes=5)

# define our full model
model = tf.keras.Sequential([
	# defining the input dimension
	Input((512, 512, 3)),

	BatchNormalization(renorm=True, trainable=False),
	base_model,
	Flatten(),
	Dense(256),

	LeakyReLU(),
	BatchNormalization(trainable=False),

	Dense(128),
	# leakyRelu as activation
	LeakyReLU(),
	BatchNormalization(trainable=False),
	# dropout to prevent overfitting
	Dropout(0.4),
	LeakyReLU(),

	BatchNormalization(trainable=False),

	Dense(64),
	LeakyReLU(),
	BatchNormalization(trainable=False),
	Dense(32),
	LeakyReLU(),
	BatchNormalization(trainable=False),
	Dropout(0.4),

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

loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2)
model.compile(
	optimizer=tf.keras.optimizers.SGD(0.03),
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
                                                batch_size=16,
                                                subset="training", shuffle=True),
                    callbacks=[early, model_checkpoint_callback],
                    epochs=10, validation_data=datagen.flow_from_dataframe(dataframe=train_csv,
                                                                           directory=r"/content/train_images",
                                                                           x_col="image_id",
                                                                           y_col="label", target_size=(512, 512),
                                                                           class_mode="categorical", batch_size=16,
                                                                           subset="validation", shuffle=True))
model.load_weights(checkpoint_filepath)
