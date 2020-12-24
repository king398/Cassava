import tensorflow as tf


def decode_image(image):
	image = tf.image.decode_jpeg(image, channels=3)
	image = tf.image.random_crop(image, size=(512, 512, 3))
	image = tf.image.random_brightness(image, 0.2)
	image = tf.cast(image, tf.float32)

	return image
