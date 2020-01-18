import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, SeparableConv2D, ReLU, add,
    BatchNormalization, DepthwiseConv2D,
    AveragePooling2D, Lambda, concatenate
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
    x = ReLU(x)
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
    x = ReLU(x)
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
    )(x)
    if residual:
        x = add()([x, input_tensor])
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
        x = Lambda(lambda x: tf.image.resize(x, (w,h)))(x)
        _list.append(x)
    x = concatenate(_list)
    return x