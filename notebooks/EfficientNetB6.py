import tensorflow as tf
from tensorflow.keras.models import Sequential
import datetime

from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPool2D, Conv2DTranspose, LeakyReLU, \
	BatchNormalization
from tensorflow.keras.mixed_precision import experimental as mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

model = Sequential()

Efficient_net = tf.keras.applications.EfficientNetB6(input_shape=(300, 300, 3), include_top=False)
model.add(tf.keras.layers.experimental.preprocessing.Rescaling(255))
model.add(Conv2DTranspose(filters=256, kernel_size=2, strides=1))
model.add(Efficient_net)
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(512))
model.add(Dense(256))

model.add(Dense(128))

model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))

model.add(Dense(8))

model.add(Dense(5, activation="softmax"))
opt = tf.keras.optimizers.SGD(learning_rate=0.05)
model.compile(optimizer=opt,
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])
checkpoint_filepath = "/content/temp"
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                               monitor='val_accuracy',
                                                               mode='max',
                                                               save_best_only=True)

model.fit(images, labels, batch_size=16
          , shuffle=True, epochs=6, callbacks=[model_checkpoint_callback, tensorboard_callback], validation_split=0.2)
