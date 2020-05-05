from functools import wraps
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from utils1.utils1 import compose


#--------------------------------------------------#
#   单次卷积
#--------------------------------------------------#
@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}  ## 正则化
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

#---------------------------------------------------#
#   卷积块
#   DarknetConv2D + BatchNormalization + LeakyReLU
#---------------------------------------------------#
def DarknetConv2D_BN_Leaky(*args, **kwargs):  ### Darknet特殊卷积快
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose( 
        DarknetConv2D(*args, **no_bias_kwargs), ##卷积 正则 激活
        BatchNormalization(),
        LeakyReLU(alpha=0.1))

#---------------------------------------------------#
#   卷积块
#   DarknetConv2D + BatchNormalization + LeakyReLU
#---------------------------------------------------#
def resblock_body(x, num_filters, num_blocks): ### 前一个特征层不经过卷积进入后面的特征层并形成映射
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3,3), strides=(2,2))(x)  ## 首先调整了输入的长和宽 因为有步长
    for i in range(num_blocks):
        y = DarknetConv2D_BN_Leaky(num_filters//2, (1,1))(x)  ## 压缩一半的通道数
        y = DarknetConv2D_BN_Leaky(num_filters, (3,3))(y)    ## 恢复原来的通道数 且大小不变
        x = Add()([x,y])
    return x

#---------------------------------------------------#
#   darknet53 的主体部分
#---------------------------------------------------#
def darknet_body(x):
    x = DarknetConv2D_BN_Leaky(32, (3,3))(x)

    # 208 208
    x = resblock_body(x, 64, 1) #输入 通道 重复次数
    # 104 104
    x = resblock_body(x, 128, 2)
    # 52 52
    x = resblock_body(x, 256, 8)
    feat1 = x
    # 26 26
    x = resblock_body(x, 512, 8)
    feat2 = x
    #13 13
    x = resblock_body(x, 1024, 4)
    feat3 = x
    return feat1,feat2,feat3

