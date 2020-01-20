import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, SeparableConv2D, ReLU, Dropout,
    BatchNormalization, DepthwiseConv2D, add, Softmax,
    AveragePooling2D, Lambda, concatenate, UpSampling2D
)
from tensorflow.keras import backend as K


def ConvBlock(input_tensor, filters, kernel_size, strides, padding='same', activation=True):
    '''Convolutional Block
    Reference: https://arxiv.org/pdf/1902.04502.pdf
    Params:
        input_tensor    -> Input Tensor
        filters         -> Number of filters
        kernel_size     -> Size of convolutional kernel
        strides         -> Strides of convolutional kernel
        padding         -> Type of padding ('same'/'valid')
        activation      -> Apply activation function (flag)
    '''
    x = Conv2D(
        filters, kernel_size,
        strides=strides, padding=padding
    )(input_tensor)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def DSConvBlock(input_tensor, filters, kernel_size, strides, padding='same', activation=True):
    '''Depthwise Seperable Convolutional Block
    Reference:
        https://arxiv.org/pdf/1902.04502.pdf
        http://geekyrakshit.com/deep-learning/depthwise-separable-convolution/
    Params:
        input_tensor    -> Input Tensor
        filters         -> Number of filters
        kernel_size     -> Size of convolutional kernel
        strides         -> Strides of convolutional kernel
        padding         -> Type of padding ('same'/'valid')
        activation      -> Apply activation function (flag)
    '''
    x = SeparableConv2D(
        filters, kernel_size,
        strides=strides, padding=padding
    )(input_tensor)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def _bottleneck(input_tensor, filters, kernel_size, t_channels, strides, residual=True):
    t_channels = K.int_shape(input_tensor)[-1] * t_channels
    x = ConvBlock(input_tensor, t_channels, (1, 1), (1, 1))
    x = DepthwiseConv2D(
        kernel_size, (strides, strides),
        depth_multiplier=1, padding='same'
    )(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = ConvBlock(
        x, filters, (1, 1),
        (1, 1), activation=False
    )
    if residual:
        x = add([x, input_tensor])
    return x


def BottleNeck(input_tensor, filters, kernel_size, t_channels, strides, n_layers):
    '''BottleNeck Block
    Reference: https://arxiv.org/pdf/1902.04502.pdf
    Params:
        input_tensor    -> Input Tensor
        filters         -> Number of filters
        kernel_size     -> Size of convolutional kernel
        t_channels      -> Number of output channels
        strides         -> Strides of convolutional kernel
        n_layers        -> Number of bottleneck layers
    '''
    x = _bottleneck(
        input_tensor, filters, kernel_size,
        t_channels, strides, False
    )
    for _ in range(1, n_layers):
        x = _bottleneck(x, filters, kernel_size, t_channels, 1)
    return x


def PPM(input_tensor, bin_sizes, height=32, width=64):
    '''Pyramid Pooling Module
    References:
        https://arxiv.org/pdf/1902.04502.pdf
        https://arxiv.org/pdf/1612.01105.pdf
    Params:
        input_tensor    -> Input Tensor
        bin_size        -> PSPNet Bin Sizes
        height          -> Image Height
        width           -> Image Width
    '''
    _list = [input_tensor]
    for size in bin_sizes:
        x = AveragePooling2D(
            pool_size=(width // size, height // size),
            strides=(width // size, height // size)
        )(input_tensor)
        x = Conv2D(128, 3, 2, padding='same')(x)
        x = Lambda(lambda x: tf.image.resize(x, (width, height)))(x)
        _list.append(x)
    x = concatenate(_list)
    return x



def FFM(downsample_layer, feat_ext_layer):
    '''Feature Fusion Module
    Reference: https://arxiv.org/pdf/1902.04502.pdf
    Params:
        downsample_layer -> Output of Downsample Layers
        feat_ext_layer   -> Output of Global Feature Extraction
    '''
    fusion_layer_1 = ConvBlock(
        downsample_layer, 128, (1, 1),
        (1, 1), 'same', False
    )
    fusion_layer_2 = UpSampling2D((4, 4))(feat_ext_layer)
    fusion_layer_2 = SeparableConv2D(
        128, (3, 3), padding='same', strides = (1, 1),
        activation=None, dilation_rate=(4, 4)
    )(fusion_layer_2)
    fusion_layer_2 = BatchNormalization()(fusion_layer_2)
    fusion_layer_2 = ReLU()(fusion_layer_2)
    fusion_layer_2 = Conv2D(
        128, 1, 1, padding='same',
        activation=None
    )(fusion_layer_2)
    fusion_layer = add([fusion_layer_1, fusion_layer_2])
    fusion_layer = BatchNormalization()(fusion_layer)
    fusion_layer = ReLU()(fusion_layer)
    return fusion_layer



def Classifier(input_tensor, n_classes=12):
    '''Feature Fusion Module
    Reference: https://arxiv.org/pdf/1902.04502.pdf
    Params:
        input_tensor -> Input Tensor
        n_classes    -> Number of output classes
    '''
    x = SeparableConv2D(128, (3, 3), padding='same', strides = (1, 1))(input_tensor)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = SeparableConv2D(128, (3, 3), padding='same', strides = (1, 1))(input_tensor)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = ConvBlock(x, n_classes, (1, 1), (1, 1), activation=False)
    x = Dropout(0.3)(x)
    x = UpSampling2D((8, 8))(x)
    x = Softmax()(x)
    return x