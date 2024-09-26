

import os

import numpy as np
import matplotlib.pyplot as plt
import tqdm as tq

from tensorflow.keras.utils import Sequence


from tensorflow import data as tf_data
from tensorflow import image as tf_image
from tensorflow import io as tf_io

import tensorflow as tf
from PIL import Image


class DataGenerator(Sequence):
    def __init__(self, data_folder, ground_truth_folder, batch_size, img_shape, shuffle=True, data_augmentation=False,max_examples=0):
        super().__init__()
        self.data_folder = data_folder
        self.ground_truth_folder = ground_truth_folder
        self.batch_size = batch_size
        self.img_shape = img_shape
        self.shuffle = shuffle
        self.data_files = sorted([f for f in os.listdir(data_folder) ])
        self.ground_truth_files = sorted([f for f in os.listdir(ground_truth_folder) ])
        self.indexes = np.arange(len(self.data_files))
        self.max_examples=max_examples
        self.on_epoch_end()
        self.rng =  tf.random.Generator.from_seed(123, alg='philox')
        self.data_augmentation = data_augmentation

    def __len__(self):
        # Number of batches per epoch
        if  self.max_examples ==0:
          num_exemp=len(self.data_files)
        else:
          num_exemp=self.max_examples
        return int(np.floor(num_exemp / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        batch_data_files = [self.data_files[k] for k in batch_indexes]
        batch_ground_truth_files = [self.ground_truth_files[k] for k in batch_indexes]

        # Generate data
        X, y = self.__data_generation(batch_data_files, batch_ground_truth_files)

        return X, y

    def on_epoch_end(self):
        # Updates indexes after each epoch
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def scale_array(self, array):
        min_val = np.min(array)
        max_val = np.max(array)
        scaled_array = 2 * (array - min_val) / (max_val - min_val) - 1
        return scaled_array


    def augment_data(self, data_array, mascara_array):

        #left right
        lr_coin = tf.less(tf.random.uniform((), 0., 1.), 0.5)
        data_array = tf.cond(lr_coin, lambda: tf_image.flip_left_right(data_array), lambda: data_array)
        mascara_array = tf.cond(lr_coin, lambda: tf_image.flip_left_right(mascara_array), lambda: mascara_array)

        # updown flip
        ud_coin = tf.less(tf.random.uniform((), 0., 1.), 0.5)
        data_array = tf.cond(ud_coin, lambda: tf_image.flip_up_down(data_array), lambda: data_array)
        mascara_array = tf.cond(ud_coin, lambda: tf_image.flip_up_down(mascara_array), lambda: mascara_array)

        # brightness
        bright_coin = tf.less(tf.random.uniform((), 0., 1.), 0.2)
        seed = self.rng.make_seeds(2)[0]
        data_array=tf.cond(bright_coin, lambda: np.array(tf_image.stateless_random_brightness(np.array(data_array).T, max_delta=0.95, seed=seed)).T, lambda: data_array)


        #contrast
        contrastt_coin = tf.less(tf.random.uniform((), 0., 1.), 0.2)
        seed = self.rng.make_seeds(2)[0]
        data_array=tf.cond(contrastt_coin, lambda: np.array(tf_image.stateless_random_contrast(np.array(data_array).T, lower=0.1, upper=0.9, seed=seed)).T, lambda: data_array)

        data_array = self.scale_array(data_array)

        return data_array, mascara_array



    def __data_generation(self, batch_data_files, batch_ground_truth_files):
        # Initialization
        X = np.empty((self.batch_size, *self.img_shape), dtype='float32')
        y = np.empty((self.batch_size, *self.img_shape), dtype='float32')

        # Generate data
        for i, (data_file, gt_file) in enumerate(zip(batch_data_files, batch_ground_truth_files)):
            # Load data and ground truth
            data_array = np.loadtxt(os.path.join(self.data_folder, data_file))
            data_array = self.scale_array(data_array)
            data_array =data_array.reshape(self.img_shape)

            ground_truth_array= np.loadtxt(os.path.join(self.ground_truth_folder, gt_file))
            ground_truth_array=ground_truth_array.reshape(self.img_shape)#np.expand_dims(ground_truth_array,axis=2)

            # data augmentation

            if self.data_augmentation:
               data_array, ground_truth_array =   self.augment_data(data_array,ground_truth_array)



            X[i,] = data_array#.reshape(self.img_shape)
            y[i,] = np.array(ground_truth_array).astype(np.uint8) #.reshape(self.img_shape)

        return X, y
