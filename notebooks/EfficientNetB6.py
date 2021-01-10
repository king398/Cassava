import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import pandas as pd
from tensorflow.keras.layers import Flatten, Dense, LeakyReLU, BatchNormalization, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)
tf.keras.regularizers.l1(l1=0.01)
datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2, horizontal_flip=True)
train_csv = pd.read_csv(r"/content/train.csv")
train_csv["label"] = train_csv["label"].astype(str)

base_model = tf.keras.applications.EfficientNetB6(include_top=False)
base_model.trainable = True

radam = tfa.optimizers.RectifiedAdam()
ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)


def log_t(u, t):
	"""Compute log_t for `u`."""
	if t == 1.0:
		return tf.math.log(u)
	else:
		return (u ** (1.0 - t) - 1.0) / (1.0 - t)


def exp_t(u, t):
	"""Compute exp_t for `u`."""
	if t == 1.0:
		return tf.math.exp(u)
	else:
		return tf.math.maximum(0, 1.0 + (1.0 - t) * u) ** (1.0 / (1.0 - t))


def compute_normalization_fixed_point(y_pred, t, num_iters=5):
	"""Returns the normalization value for each example (t > 1.0).
	  Args:
	  y_pred: A multi-dimensional tensor with last dimension `num_classes`.
	  t: Temperature 2 (> 1.0 for tail heaviness).
	  num_iters: Number of iterations to run the method.
	  Return: A tensor of same rank as y_pred with the last dimension being 1.
	"""
	mu = tf.math.reduce_max(y_pred, -1, keepdims=True)
	normalized_y_pred_step_0 = y_pred - mu
	normalized_y_pred = normalized_y_pred_step_0
	i = 0
	while i < num_iters:
		i += 1
		logt_partition = tf.math.reduce_sum(exp_t(normalized_y_pred, t), -1, keepdims=True)
		normalized_y_pred = normalized_y_pred_step_0 * (logt_partition ** (1.0 - t))

	logt_partition = tf.math.reduce_sum(exp_t(normalized_y_pred, t), -1, keepdims=True)
	return -log_t(1.0 / logt_partition, t) + mu


def compute_normalization(y_pred, t, num_iters=5):
	"""Returns the normalization value for each example.
	  Args:
	  y_pred: A multi-dimensional tensor with last dimension `num_classes`.
	  t: Temperature 2 (< 1.0 for finite support, > 1.0 for tail heaviness).
	  num_iters: Number of iterations to run the method.
	  Return: A tensor of same rank as activation with the last dimension being 1.
	"""
	if t < 1.0:
		return None  # not implemented as these values do not occur in the authors experiments...
	else:
		return compute_normalization_fixed_point(y_pred, t, num_iters)


def tempered_softmax(y_pred, t, num_iters=5):
	"""Tempered softmax function.
	  Args:
	  y_pred: A multi-dimensional tensor with last dimension `num_classes`.
	  t: Temperature tensor > 0.0.
	  num_iters: Number of iterations to run the method.
	  Returns:
	  A probabilities tensor.
	"""
	if t == 1.0:
		normalization_constants = tf.math.log(tf.math.reduce_sum(tf.math.exp(y_pred), -1, keepdims=True))
	else:
		normalization_constants = compute_normalization(y_pred, t, num_iters)

	return exp_t(y_pred - normalization_constants, t)


def bi_tempered_logistic_loss(y_pred, y_true, t1, t2, num_iters=5):
	"""Bi-Tempered Logistic Loss with custom gradient.
	  Args:
	  y_pred: A multi-dimensional tensor with last dimension `num_classes`.
	  y_true: A tensor with shape and dtype as y_pred.
	  t1: Temperature 1 (< 1.0 for boundedness).
	  t2: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
	  num_iters: Number of iterations to run the method.
	  Returns:
	  A loss tensor.
	"""
	probabilities = tempered_softmax(y_pred, t2, num_iters)

	temp1 = (log_t(y_true + 1e-10, t1) - log_t(probabilities, t1)) * y_true
	temp2 = (1 / (2 - t1)) * (tf.math.pow(y_true, 2 - t1) - tf.math.pow(probabilities, 2 - t1))
	loss_values = temp1 - temp2

	return tf.math.reduce_sum(loss_values, -1)


class BiTemperedLogisticLoss(tf.keras.losses.Loss):
	def __init__(self, t1, t2, n_iter=5):
		super(BiTemperedLogisticLoss, self).__init__()
		self.t1 = t1
		self.t2 = t2
		self.n_iter = n_iter

	def call(self, y_true, y_pred):
		return bi_tempered_logistic_loss(y_pred, y_true, self.t1, self.t2, self.n_iter)


model = tf.keras.Sequential([
	tf.keras.layers.Input((512, 512, 3)),
	tf.keras.layers.BatchNormalization(renorm=True),
	base_model,
	tf.keras.layers.LeakyReLU(),
	BatchNormalization(trainable=False),
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(256),
	tf.keras.layers.LeakyReLU(),
	BatchNormalization(trainable=False),

	tf.keras.layers.Dense(128),
	tf.keras.layers.LeakyReLU(),

	BatchNormalization(trainable=False),

	tf.keras.layers.LeakyReLU(),

	tf.keras.layers.Dropout(0.4),
	tf.keras.layers.LeakyReLU(),

	BatchNormalization(trainable=False),

	tf.keras.layers.Dense(64),

	tf.keras.layers.LeakyReLU(),
	BatchNormalization(trainable=False),

	tf.keras.layers.Dense(32),
	tf.keras.layers.LeakyReLU(),

	BatchNormalization(trainable=False),

	tf.keras.layers.Dropout(0.4),

	tf.keras.layers.LeakyReLU(),
	BatchNormalization(trainable=False),

	tf.keras.layers.Dense(16),
	tf.keras.layers.LeakyReLU(),
	BatchNormalization(trainable=False),
	tf.keras.layers.Dense(8),
	tf.keras.layers.LeakyReLU(),
	BatchNormalization(trainable=False),
	tf.keras.layers.Dense(5, activation='softmax')
])
opt = tf.keras.optimizers.RMSprop(learning_rate=0.1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=3, min_lr=0.001)
model.compile(
	optimizer=opt,
	loss=BiTemperedLogisticLoss(t1=0, t2=1.0),
	metrics=['categorical_accuracy'])

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
model.fit(datagen.flow_from_dataframe(dataframe=train_csv,
                                      directory=r"/content/train_images", x_col="image_id",
                                      y_col="label", target_size=(512, 512), class_mode="categorical", batch_size=8,
                                      subset="training", shuffle=True),
          callbacks=[early, model_checkpoint_callback, reduce_lr],
          epochs=10, validation_data=datagen.flow_from_dataframe(dataframe=train_csv,
                                                                 directory=r"/content/train_images",
                                                                 x_col="image_id",
                                                                 y_col="label", target_size=(512, 512),
                                                                 class_mode="categorical", batch_size=8,
                                                                 subset="validation", shuffle=True), batch_size=8)
model.load_weights(checkpoint_filepath)
