from os import name
from numpy.lib.arraypad import pad
from tensorflow.keras import models
from tensorflow.keras.layers import *
from tensorflow.python.keras.backend import shape


class Mobilenet_V2():
    def __init__(self, *, inp_shape = (256, 384,1), rho = 1.0 , alpha = 1.0, expansion = 6.0, classes = 2, droppout = 0.0):
        assert alpha > 0 and alpha <= 1 ,'Error, my Mobilenet_V2 can only accept  alpha > 0 and alpha <= 1'
        assert rho > 0 and rho <= 1 ,'Error, my Mobilenet_V2 can only accept  rho > 0 and rho <= 1'
        self._inp_shape = inp_shape
        self._rho = rho
        self._alpha = alpha
        self._expansion = expansion
        self._classes = classes
        self._droppout = droppout
    def _depthwiseconv(self, *, strides: int):
        return models.Sequential([
            DepthwiseConv2D(kernel_size= (3,3), strides= strides, padding= 'same' if strides == 1 else 'valid', use_bias= False),
            BatchNormalization(),
            ReLU(max_value= 6.)
        ])
    def _pointwiseconv(self, *, filters: int, linear: bool):
        layer = models.Sequential([
            Conv2D(filters= int(filters * self._alpha), kernel_size= (1,1), strides= (1,1), padding= 'same', use_bias= False),
            BatchNormalization(),
        ])
        if linear == False:
            layer.add(ReLU(max_value= 6.))
        return layer
    def _standardconv(self):
        return models.Sequential([
            Conv2D(filters= 32, kernel_size= (3,3), strides= (2,2), use_bias= False),
            BatchNormalization(),
            ReLU(max_value= 6.)
        ])
    def _inverted_residual_block_(self, x, *, strides_depthwise: int, filter_pointwise: int, expansion: int):
        filter = int(filter_pointwise * self._alpha)
        fx = self._pointwiseconv(filters= filter * expansion, linear= False)(x)
        fx = self._depthwiseconv(strides= strides_depthwise)(fx)
        fx = self._pointwiseconv(filters= filter , linear= True)(fx)
        if strides_depthwise == 1 and x.shape[-1] == filter_pointwise:
            return add([fx,x])
        else:
            return fx
    def _bottleneck_block_(self, x, *,  s: int, c: int, t: int, n: int):
        '''
            s : strides
            c : output channels
            t : expansion factor
            n : repeat
        '''
        x = self._inverted_residual_block_(x, strides_depthwise= s, filter_pointwise= c, expansion= t)
        for i in range(n-1):
            x = self._inverted_residual_block_(x, strides_depthwise= 1, filter_pointwise= c, expansion= t)
        return x
    def build(self,x):
        #feature_map = int(self._rho * self._inp_shape[0])
        #feature_map2 = int(self._rho * self._inp_shape[1])
        #img_inp = Input(shape= (feature_map, feature_map2,1))
        # standardconv
        x = self._standardconv()(x)
        # block bottleneck 1
        x = self._bottleneck_block_(x, s= 1, c= 16, t= 1, n= 1)
        # block bottleneck 2
        x = self._bottleneck_block_(x, s= 2, c= 24, t= self._expansion, n= 2)
        # block bottleneck 3
        x = self._bottleneck_block_(x, s= 2, c= 32, t= self._expansion, n= 3)
        # block bottleneck 4
        x = self._bottleneck_block_(x, s= 2, c= 64, t= self._expansion, n= 4)
        # block bottleneck 5
        x = self._bottleneck_block_(x, s= 1, c= 96, t= self._expansion, n= 3)
        # block bottleneck 6
        x = self._bottleneck_block_(x, s= 2, c= 160, t= self._expansion, n= 3)
        # block bottleneck 7
        x = self._bottleneck_block_(x, s= 1, c= 320, t= self._expansion, n= 1)
        # conv2d 1x1
        x = self._pointwiseconv(filters= 1280, linear= False)(x)
        # fully connect
        #x = GlobalAveragePooling2D()(x)
        #x = Dropout(self._droppout)(x)
        #x = Dense(self._classes, activation='softmax')(x)
        return x


input_shape= (256, 384,1)

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

def build_model_v2(input_shape):

  input_shape=(input_shape[0], input_shape[1], 1)
  inputs = Input(input_shape)


  encoder= Mobilenet_V2().build(inputs)
  decoder = create_decoder(encoder, transpose=False)
  outputs = Conv2D(1, (1, 1), activation='sigmoid')(decoder)

  return  models.Model(inputs, outputs)

#model= build_model_v2(input_shape)
#model.summary()