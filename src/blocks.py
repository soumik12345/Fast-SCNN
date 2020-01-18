from tensorflow.keras.layers import (
    Conv2D, SeparableConv2D,
    BatchNormalization, ReLU
)


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