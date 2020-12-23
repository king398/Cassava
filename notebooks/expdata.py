import math, re, os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from functools import partial
from sklearn.model_selection import train_test_split
import tensorflow.keras.mixed_precision as mixed_precision

print("Tensorflow version " + tf.__version__)
AUTOTUNE = tf.data.experimental.AUTOTUNE
IMAGE_SIZE = [512, 512]
CLASSES = ['0', '1', '2', '3', '4']


def decode_image(image):
	image = tf.image.decode_jpeg(image, channels=3)
	image = tf.cast(image, tf.float32) / 255.0
	image = tf.reshape(image, [*IMAGE_SIZE, 3])
	return image


def read_tfrecord(example, labeled):
	tfrecord_format = {
		"image": tf.io.FixedLenFeature([], tf.string),
		"target": tf.io.FixedLenFeature([], tf.int64)
	} if labeled else {
		"image": tf.io.FixedLenFeature([], tf.string),
		"image_name": tf.io.FixedLenFeature([], tf.string)
	}
	example = tf.io.parse_single_example(example, tfrecord_format)
	image = decode_image(example['image'])
	if labeled:
		label = tf.cast(example['target'], tf.int32)
		return image, label
	idnum = example['image_name']
	return image, idnum


def load_dataset(filenames, labeled=True, ordered=False):
	ignore_order = tf.data.Options()
	if not ordered:
		ignore_order.experimental_deterministic = False  # disable order, increase speed
	dataset = tf.data.TFRecordDataset(filenames,
	                                  num_parallel_reads=AUTOTUNE)  # automatically interleaves reads from multiple files
	dataset = dataset.with_options(
		ignore_order)  # uses data as soon as it streams in, rather than in its original order
	dataset = dataset.map(partial(read_tfrecord, labeled=labeled), num_parallel_calls=AUTOTUNE)
	return dataset


tfrec = []
for i in os.listdir(r"/content/train_tfrecords"):
	tfrec.append(os.path.join("/content/train_tfrecords", i))


def data_augment(image, label):
	# Thanks to the dataset.prefetch(AUTO) statement in the following function this happens essentially for free on TPU.
	# Data pipeline code is executed on the "CPU" part of the TPU while the TPU itself is computing gradients.
	image = tf.image.random_flip_left_right(image)
	return image, label


def get_training_dataset():
	dataset = load_dataset(tfrec, labeled=True)
	dataset = dataset.map(data_augment, num_parallel_calls=AUTOTUNE)
	dataset = dataset.repeat()
	dataset = dataset.shuffle(2048)
	dataset = dataset.batch(32)
	dataset = dataset.prefetch(AUTOTUNE)
	return dataset


print("Training data shapes:")
for image, label in get_training_dataset().take(3):
	print(image.numpy().shape, label.numpy().shape)
	print("Training data label examples:", label.numpy())
	print("Validation data shapes:")
