from keras import layers, models, regularizers, initializers
import tensorflow as tf

def get_model_basic_seg(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    dropout = 0.3
    L2 = 0.1
    initializer = initializers.HeNormal()

    # Encoder
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(L2),
                      kernel_initializer=initializer)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout)(x)


    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(L2),
                      kernel_initializer=initializer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Dropout(dropout)(x)

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(L2),
                      kernel_initializer=initializer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(L2),
                      kernel_initializer=initializer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout)(x)

    # Bottleneck
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(L2),
                      kernel_initializer=initializer)(x)

    # Decoder
    x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same',
                               activation='relu',
                               kernel_regularizer=regularizers.l2(L2),
                               kernel_initializer=initializer)(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(L2),
                      kernel_initializer=initializer)(x)

    x = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',
                               activation='relu',
                               kernel_regularizer=regularizers.l2(L2),
                               kernel_initializer=initializer)(x)

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(L2),
                      kernel_initializer=initializer)(x)

    x = layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same',
                               activation='relu',
                               kernel_regularizer=regularizers.l2(L2),
                               kernel_initializer=initializer)(x)

    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(L2),
                      kernel_initializer=initializer)(x)

    x = layers.Conv2DTranspose(8, (3, 3), strides=(2, 2), padding='same',
                               activation='relu',
                               kernel_regularizer=regularizers.l2(L2),
                               kernel_initializer=initializer)(x)

    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(L2),
                      kernel_initializer=initializer)(x)

    # Output layer
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(x)

    # Create the model
    model = models.Model(inputs, outputs)
    return model
