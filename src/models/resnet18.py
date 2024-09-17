import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Lambda, Dropout
from tensorflow.keras.models import Model
from keras_cv.models import ResNet18Backbone

def resnet_segmentation_model18(input_shape=(512, 512, 1)):


    inputs = Input(input_shape)
    dropout= 0.3

    #x = Lambda(lambda x: tf.repeat(x, repeats=3, axis=-1))(inputs)
    x = Conv2D(3, (1, 1), padding='same',use_bias=False, activation='relu')(inputs)
    
    base_model =  ResNet18Backbone()


    base_output = base_model(x)

    x = UpSampling2D((2, 2), interpolation="bilinear")(base_output)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = Dropout(0.5)(x)  # adjust dropout as needed

    # Upsample from 16x16 to 32x32
    x = UpSampling2D((2, 2), interpolation="bilinear")(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = Dropout(0.5)(x)

    # Upsample from 32x32 to 64x64
    x = UpSampling2D((2, 2), interpolation="bilinear")(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = Dropout(0.5)(x)

    # Upsample from 64x64 to 128x128
    x = UpSampling2D((2, 2), interpolation="bilinear")(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = Dropout(0.5)(x)

    # Upsample from 128x128 to 256x256
    x = UpSampling2D((2, 2), interpolation="bilinear")(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = Dropout(0.5)(x)


    outputs = Conv2D(1, (1, 1), activation='sigmoid')(x)

    model = Model(inputs, outputs)
    return model

model = resnet_segmentation_model18((256,384,1))
model.summary()
