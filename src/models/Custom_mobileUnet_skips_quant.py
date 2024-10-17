import tempfile
import os

import tensorflow as tf

from tensorflow_model_optimization.python.core.keras.compat import keras


def mobileunet(pretrained_weights = None,input_size = (256,384,1)):
    inputs = keras.layers.Input(input_size)

    conv1  = keras.layers.SeparableConv2D(32, 3, activation='relu', padding='same')(inputs)
    conv1  = keras.layers.BatchNormalization()(conv1)
    conv1  = keras.layers.SeparableConv2D(32, 3, activation='relu', padding='same')(conv1)
    conv1  = keras.layers.BatchNormalization()(conv1)
    pool1  = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2  = keras.layers.SeparableConv2D(64, 3, activation='relu', padding='same')(pool1)
    conv2  = keras.layers.BatchNormalization()(conv2)
    conv2  = keras.layers.SeparableConv2D(64, 3, activation='relu', padding='same')(conv2)
    conv2  = keras.layers.BatchNormalization()(conv2)
    pool2  = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3  = keras.layers.SeparableConv2D(128, 3, activation='relu', padding='same')(pool2)
    conv3  = keras.layers.BatchNormalization()(conv3)
    conv3  = keras.layers.SeparableConv2D(128, 3, activation='relu', padding='same')(conv3)
    conv3  = keras.layers.BatchNormalization()(conv3)
    pool3  = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4  = keras.layers.SeparableConv2D(256, 3, activation='relu', padding='same')(pool3)
    conv4  = keras.layers.BatchNormalization()(conv4)
    conv4  = keras.layers.SeparableConv2D(256, 3, activation='relu', padding='same')(conv4)
    conv4  = keras.layers.BatchNormalization()(conv4)
    pool4  = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5  = keras.layers.SeparableConv2D(512, 3, activation='relu', padding='same')(pool4)
    conv5  = keras.layers.BatchNormalization()(conv5)
    conv5  = keras.layers.SeparableConv2D(512, 3, activation='relu', padding='same')(conv5)
    conv5  = keras.layers.BatchNormalization()(conv5)

    conv6  = keras.layers.Conv2DTranspose(256, 3, strides=(2, 2), activation='relu', padding='same')(conv5)
    cat6   = keras.layers.concatenate([conv4, conv6], axis = 3)
    conv6  = keras.layers.SeparableConv2D(256, 3, activation='relu', padding='same')(cat6)
    conv6  = keras.layers.BatchNormalization()(conv6)
    conv6  = keras.layers.SeparableConv2D(256, 3, activation='relu', padding='same')(conv6)
    conv6  = keras.layers.BatchNormalization()(conv6)

    conv7  = keras.layers.Conv2DTranspose(128, 3, strides=(2, 2), activation='relu', padding='same')(conv6)
    cat7   = keras.layers.concatenate([conv3, conv7], axis = 3)
    conv7  = keras.layers.SeparableConv2D(128, 3, activation='relu', padding='same')(cat7)
    conv7  = keras.layers.BatchNormalization()(conv7)
    conv7  = keras.layers.SeparableConv2D(128, 3, activation='relu', padding='same')(conv7)
    conv7  = keras.layers.BatchNormalization()(conv7)

    conv8  = keras.layers.Conv2DTranspose(64, 3, strides=(2, 2), activation='relu', padding='same')(conv7)
    cat8   = keras.layers.concatenate([conv2, conv8], axis = 3)
    conv8  = keras.layers.SeparableConv2D(64, 3, activation='relu', padding='same')(cat8)
    conv8  = keras.layers.BatchNormalization()(conv8)
    conv8  = keras.layers.SeparableConv2D(64, 3, activation='relu', padding='same')(conv8)
    conv8  = keras.layers.BatchNormalization()(conv8)

    conv9  = keras.layers.Conv2DTranspose(32, 3, strides=(2, 2), activation='relu', padding='same')(conv8)
    cat9   = keras.layers.concatenate([conv1, conv9], axis = 3)
    conv9  = keras.layers.SeparableConv2D(32, 3, activation='relu', padding='same')(cat9)
    conv9  = keras.layers.BatchNormalization()(conv9)
    conv9  = keras.layers.SeparableConv2D(32, 3, activation='relu', padding='same')(conv9)
    conv9  = keras.layers.BatchNormalization()(conv9)
    conv9  = keras.layers.Conv2D(2, 3, activation='relu', padding='same')(conv9)
    conv10 = keras.layers.Conv2D(1, 1, activation='sigmoid')(conv9)

    model = keras.models.Model( inputs,  conv10)
    #model.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy', metrics=['accuracy'])

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model
