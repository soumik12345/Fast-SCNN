from .blocks import *
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model


def FastSCNN(input_shape=(2048, 1024, 3)):
    '''Fast-SCNN Model
    Reference: https://arxiv.org/pdf/1902.04502.pdf
    Params:
        input_shape -> Shape of input image
    '''
    model_input = Input(input_shape, name='model_input')
    # Downsample Layers
    downsample_layer = ConvBlock(model_input, 32, (3, 3), (2, 2))
    downsample_layer = DSConvBlock(downsample_layer, 48, (3, 3), (2, 2))
    downsample_layer = DSConvBlock(downsample_layer, 64, (3, 3), (2, 2))
    # Global Feature Extraction
    feat_ext_layer = BottleNeck(downsample_layer, 64, (3, 3), 6, 2, 3)
    feat_ext_layer = BottleNeck(feat_ext_layer, 96, (3, 3), 6, 2, 3)
    feat_ext_layer = BottleNeck(feat_ext_layer, 128, (3, 3), 6, 1, 3)
    feat_ext_layer = PPM(feat_ext_layer, [2, 4, 6, 8])
    return Model(model_input, feat_ext_layer, name='Fast-SCNN')
