import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l1

policy = mixed_precision.Policy('mixed_bfloat16')
mixed_precision.set_policy(policy)
reg = l1(0.001)

with strategy.scope():
	base_model = tf.keras.applications.EfficientNetB4(weights="imagenet", include_top=False)
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
		tf.keras.layers.Dense(len(CLASSES), activation='softmax')
	])

	model.compile(
		optimizer=tf.keras.optimizers.SGD(lr=0.03),
		loss='sparse_categorical_crossentropy',
		metrics=['sparse_categorical_accuracy'])

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

early = EarlyStopping(monitor='val_loss',
                      mode='min',
                      patience=5)
STEPS_PER_EPOCH = 17118 // BATCH_SIZE
VALID_STEPS = 4279 // BATCH_SIZE
checkpoint_filepath = '/tmp/checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

history = model.fit(train_dataset,
                    steps_per_epoch=STEPS_PER_EPOCH,

                    epochs=25,
                    validation_data=valid_dataset,
                    validation_steps=VALID_STEPS, callbacks=[early,model_checkpoint_callback])
model.load_weights(checkpoint_filepath)
