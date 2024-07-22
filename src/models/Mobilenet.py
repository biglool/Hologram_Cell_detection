from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, BatchNormalization, ReLU, ZeroPadding2D, UpSampling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import resnet18


def create_mobilenet_encoder(input):
    #extra layer to simulate 3 channels ( testejar)
    x = Conv2D(3, (1, 1), padding='same',use_bias=False, activation='relu')(input)

    # Define the MobileNet architecture
    x = Conv2D(32, (3, 3), (2,2), padding='same', use_bias=False, name='conv1')(x)
    x = BatchNormalization(name='conv1_bn')(x)
    x = ReLU(6., name='conv1_relu')(x)

    def depthwise_conv_block(inputs, pointwise_conv_filters, block_id, strides=(1, 1)):
        x = DepthwiseConv2D((3, 3),  padding='same' if strides == (1, 1) else 'valid', strides=strides, use_bias=False, name=f'conv_dw_{block_id}')(inputs)
        x = BatchNormalization(name=f'conv_dw_{block_id}_bn')(x)
        x = ReLU(6., name=f'conv_dw_{block_id}_relu')(x)

        x = Conv2D(pointwise_conv_filters, (1, 1), padding='same', use_bias=False, name=f'conv_pw_{block_id}')(x)
        x = BatchNormalization(name=f'conv_pw_{block_id}_bn')(x)
        x = ReLU(6., name=f'conv_pw_{block_id}_relu')(x)

        return x

    x = depthwise_conv_block(x, 64, block_id=1)
    x = ZeroPadding2D(padding=((0, 1), (0, 1)), name='conv_pad_2')(x)
    x = depthwise_conv_block(x, 128, block_id=2, strides=(2, 2))

    x = depthwise_conv_block(x, 128, block_id=3)
    x = ZeroPadding2D(padding=((0, 1), (0, 1)), name='conv_pad_4')(x)
    x = depthwise_conv_block(x, 256, block_id=4, strides=(2, 2))

    x = depthwise_conv_block(x, 256, block_id=5)
    x = ZeroPadding2D(padding=((0, 1), (0, 1)), name='conv_pad_6')(x)
    x = depthwise_conv_block(x, 512, block_id=6, strides=(2, 2))

    for i in range(7, 12):
        x = depthwise_conv_block(x, 512, block_id=i)

    x = ZeroPadding2D(padding=((0, 1), (0, 1)), name='conv_pad_12')(x)
    x = depthwise_conv_block(x, 1024, block_id=12, strides=(2, 2))

    x = depthwise_conv_block(x, 1024, block_id=13)

    return x


def create_decoder(encoder_output, transpose=True, dropout=0.1):

    # Define the decoder blocks
    def create_decoder_block(input,filters, transpose=True, dropout=0.1, name=""):

        if transpose:
            x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same', name=name+"_trans")(input)
        else:
            x = UpSampling2D((2, 2), interpolation="bilinear")(input)

        x = Conv2D(filters, (3, 3), padding='same', activation='relu',name=name)(x)
        x = Dropout(dropout)(x)  # adjust dropout as needed
        return x

    # Define the decoder architecture
    x=create_decoder_block(encoder_output,512, transpose=transpose, dropout=dropout)

    x=create_decoder_block(x ,256 ,transpose=transpose, dropout=dropout, name='dec_conv_1')
    x=create_decoder_block(x ,128 ,transpose=transpose, dropout=dropout, name='dec_conv_2')
    x=create_decoder_block(x ,64 , transpose=transpose, dropout=dropout, name='dec_conv_3')
    x=create_decoder_block(x ,32 , transpose=transpose, dropout=dropout, name='dec_conv_4')

    return x

# Create the model
def build_model(input_shape, load_weights=True):

  input_shape=(input_shape[0], input_shape[1], 1)
  inputs = Input(input_shape)

  encoder = create_mobilenet_encoder(inputs)
  decoder = create_decoder(encoder, transpose=False)
  outputs = Conv2D(1, (1, 1), activation='sigmoid')(decoder)

  model = Model(inputs, outputs)

  if load_weights:
    # Load the pre-trained MobileNet model from Keras applications with include_top=False
    pretrained_mobilenet = MobileNet(include_top=False, weights='imagenet', input_shape=(256, 384, 3))
    for layer in model.layers:
      if layer.name in [l.name for l in pretrained_mobilenet.layers]:
          #print(f"Copying weights from {layer.name} to {model.get_layer(name=layer.name).name}")
          layer.set_weights(pretrained_mobilenet.get_layer(name=layer.name).get_weights())


  return model

input_shape= (256, 384)
model =build_model(input_shape)
model.summary()