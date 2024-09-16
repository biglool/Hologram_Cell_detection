
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose,Lambda, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50

def resnet_segmentation_model(input_shape=(512, 512, 1)):


    inputs = Input(input_shape)
    dropout= 0.2

    #x = Lambda(lambda x: tf.repeat(x, repeats=3, axis=-1))(inputs)
    x = Conv2D(3, (1, 1), padding='same',use_bias=False, activation='relu')(inputs)
    
    base_model = ResNet50(include_top=False,  weights="imagenet")
    base_output = base_model(x)

    x = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(base_output)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = Dropout(0.5)(x)  # adjust dropout as needed

    # Upsample from 16x16 to 32x32
    x = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = Dropout(0.5)(x)

    # Upsample from 32x32 to 64x64
    x = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = Dropout(0.5)(x)

    # Upsample from 64x64 to 128x128
    x = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = Dropout(0.5)(x)

    # Upsample from 128x128 to 256x256
    x = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = Dropout(0.5)(x)


    outputs = Conv2D(1, (1, 1), activation='sigmoid')(x)

    model = Model(inputs, outputs)
    return model

model = resnet_segmentation_model((256,256,1))
model.summary()
