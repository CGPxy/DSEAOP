#coding=utf-8
import matplotlib

import matplotlib.pyplot as plt
import argparse
import numpy as np  
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.preprocessing import LabelEncoder
from PIL import Image  
import matplotlib.pyplot as plt  
import cv2
import random
import os
# from keras.utils import multi_gpu_model
from tqdm import tqdm  
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import tensorflow as tf

K.set_image_data_format('channels_last')  # TF dimension ordering in this code
kinit = 'glorot_normal'



def squeeze_excitation_layer(data, skipdata, out_dim):
    '''
    SE module performs inter-channel weighting.
    '''
    concatenate = Concatenate()([data, skipdata])
    concatenate = Conv2D(out_dim, (3, 3), padding="same")(concatenate)
    concatenate = BatchNormalization()(concatenate)
    concatenate = LeakyReLU(alpha=0.01)(concatenate)

    squeeze = GlobalAveragePooling2D()(concatenate)
    excitation = Dense(units=out_dim // 4)(squeeze)
    # excitation = Dense(units=out_dim)(squeeze)
    excitation = Activation('relu')(excitation)
    excitation = Dense(units=out_dim)(excitation)
    excitation = Activation('sigmoid')(excitation)
    excitation = Reshape((1, 1, out_dim))(excitation)

    scale = multiply([excitation, concatenate]) # concatenate

    data_scale = Concatenate()([data, scale])

    return data_scale



def sout(data,name):
    excitation0 = Conv2D(1, (1, 1), strides=(1, 1), padding='same')(data)
    excitation0 = Activation('sigmoid')(excitation0)
    shape_g = K.int_shape(excitation0)
    excitation0 = UpSampling2D(size=(256// shape_g[1], 256 // shape_g[2]),name=name)(excitation0)

    return excitation0


def updata(filte, data, skipdata):
    shape_x = K.int_shape(skipdata)
    shape_g = K.int_shape(data)

    up1 = UpSampling2D(size=(shape_x[1] // shape_g[1], shape_x[2] // shape_g[2]))(data)

    Selective_data = squeeze_excitation_layer(up1, skipdata, filte)

    LeakyReLU2 = ConvBlock(Selective_data, filte)
    return LeakyReLU2

def ConvBlock(data, filte):
    conv1 = Conv2D(filte, (3, 3), padding="same")(data) #,dilation_rate=(4,4)
    batch1 = BatchNormalization()(conv1)
    LeakyReLU1 = LeakyReLU(alpha=0.01)(batch1)
    conv2 = Conv2D(filte, (3, 3), padding="same")(LeakyReLU1)
    batch2 = BatchNormalization()(conv2)
    LeakyReLU2 = LeakyReLU(alpha=0.01)(batch2)
    return LeakyReLU2


def DSEM(input_size):   
    inputs = Input(shape=input_size)

    Conv1 = ConvBlock(data=inputs, filte=64)

    pool1 = MaxPooling2D(pool_size=(2, 2))(Conv1)
    Conv2 = ConvBlock(data=pool1, filte=128)

    pool2 = MaxPooling2D(pool_size=(2, 2))(Conv2)
    Conv3 = ConvBlock(data=pool2, filte=128)

    pool3 = MaxPooling2D(pool_size=(2, 2))(Conv3)   
    Conv4 = ConvBlock(data=pool3, filte=256)

    pool4 = MaxPooling2D(pool_size=(2, 2))(Conv4)    
    Conv5 = ConvBlock(data=pool4, filte=256)

    pool5 = MaxPooling2D(pool_size=(2, 2))(Conv5)    
    Conv6 = ConvBlock(data=pool5, filte=512)

    pool6 = MaxPooling2D(pool_size=(2, 2))(Conv6)    
    Conv7 = ConvBlock(data=pool6, filte=512)

    pool7 = MaxPooling2D(pool_size=(2, 2))(Conv7)    
    Conv8 = ConvBlock(data=pool7, filte=1024)

    # 6
    up1 = updata(filte=512, data=Conv8, skipdata=Conv7)
    excitation1 = sout(data=up1,name='excitation1')
    # 12
    up2 = updata(filte=512, data=up1, skipdata=Conv6)
    excitation2 = sout(data=up2,name='excitation2')
    # 25
    up3 = updata(filte=256, data=up2, skipdata=Conv5)
    excitation3 = sout(data=up3,name='excitation3')
    # 48
    up4 = updata(filte=256, data=up3, skipdata=Conv4)
    excitation4 = sout(data=up4,name='excitation4')
    # 96
    up5 = updata(filte=128, data=up4, skipdata=Conv3)
    excitation5 = sout(data=up5,name='excitation5')
    # 192
    up6 = updata(filte=128, data=up5, skipdata=Conv2)
    excitation6 = sout(data=up6,name='excitation6')
    # 384
    up7 = updata(filte=64, data=up6, skipdata=Conv1)
    excitation7 = sout(data=up7,name='excitation7')

    outconv = Conv2D(1, (1, 1), strides=(1, 1), padding='same')(up7)
    out1 = Activation('sigmoid',name='out1')(outconv)

    model = Model(inputs=inputs, outputs=[out1,excitation7,excitation6,excitation5,excitation4,excitation3,excitation2,excitation1])
    return model