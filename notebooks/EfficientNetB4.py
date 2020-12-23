train_dataset = get_training_dataset()
STEPS_PER_EPOCH = 21397 // 32
import datetime

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, LeakyReLU, \
	BatchNormalization
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.models import Sequential

model = Sequential()

Efficient_net = tf.keras.applications.EfficientNetB4(include_top=False)
Efficient_net.trainable = False
model.add(tf.keras.layers.experimental.preprocessing.Normalization())
model.add(Efficient_net)
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(Flatten())
model.add(LeakyReLU())
model.add(Dense(512))
model.add(LeakyReLU())
model.add(Dense(256))
model.add(LeakyReLU())

model.add(Dense(128))
model.add(LeakyReLU())

model.add(Dense(64))
model.add(LeakyReLU())

model.add(Dense(32))
model.add(LeakyReLU())

model.add(Dense(16))

model.add(LeakyReLU())

model.add(Dense(8))

model.add(Dense(5, activation="softmax"))
opt = tf.keras.optimizers.SGD(learning_rate=0.03, momentum=0.01)
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0.01,
                                               name='categorical_crossentropy')
model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=['categorical_accuracy'])
checkpoint_filepath = "/content/temp"
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                               monitor='val_accuracy',
                                                               mode='max',
                                                               save_best_only=True)

model.fit(train_dataset, shuffle=True, epochs=6, callbacks=[model_checkpoint_callback, tensorboard_callback],
          steps_per_epoch=STEPS_PER_EPOCH)
model = tf.keras.models.load_model(checkpoint_filepath)
model.save(r"/content/drive/MyDrive/project/effiecnetb4.h5", include_optimizer=True)
