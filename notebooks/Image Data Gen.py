import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import pandas as pd

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)
datagen = ImageDataGenerator(rescale=1. / 255)
train_csv = pd.read_csv(r"F:\Pycharm_projects\Kaggle Cassava\data\train.csv")
train = datagen.flow_from_dataframe(dataframe=train_csv,
                                    directory=r"F:\Pycharm_projects\Kaggle Cassava\data\train_images",x_col="image_id",y_col="label")
