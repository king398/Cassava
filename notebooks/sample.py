import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import pandas as pd
from tensorflow.keras.layers import Flatten, Dense, LeakyReLU, BatchNormalization, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow_addons as tfa


datagen = ImageDataGenerator(validation_split=0.2,

                              horizontal_flip=True, rescale=1. / 255)
train_csv = pd.read_csv(r"/content/train.csv")
train_csv["label"] = train_csv["label"].astype(str)
base_model = tf.keras.applications.EfficientNetB5(include_top=False, weights="imagenet", classes=5)
base_model.trainable = True

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
radam = tfa.optimizers.RectifiedAdam()
ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
model.compile(
    optimizer=ranger,
    loss=loss,
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
                                                y_col="label", target_size=(512, 512),
                                                batch_size=6,
                                                subset="training", shuffle=True),
                    callbacks=[early, model_checkpoint_callback],
                    epochs=10, validation_data=datagen.flow_from_dataframe(dataframe=train_csv,
                                                                           directory=r"/content/train_images",
                                                                           x_col="image_id",
                                                                           y_col="label", target_size=(512, 512),
                                                                           batch_size=6,
                                                                           subset="validation", shuffle=True))
model.load_weights(checkpoint_filepath)
