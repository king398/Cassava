import tensorflow as tf


def decode_image(image):
	image = tf.image.decode_jpeg(image, channels=3)
	image = tf.image.resize_images(image, size=(512, 512))
	image = tf.cast(image, tf.float32) / 255.0
	return image
