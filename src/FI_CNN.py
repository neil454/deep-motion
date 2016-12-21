from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D, BatchNormalization
from keras.models import Sequential, Model

def FI_CNN_model(num_channels):
    input_img = Input(name="input", shape=(num_channels*2, 320, 640))

    conv1 = Convolution2D(name="conv1", nb_filter=32, nb_row=3, nb_col=3, activation='relu', border_mode='same')(input_img)
    pool1 = MaxPooling2D(name="pool1", pool_size=(2, 2), border_mode='same')(conv1)
    conv2 = Convolution2D(name="conv2", nb_filter=32, nb_row=3, nb_col=3, activation='relu', border_mode='same')(pool1)
    pool2 = MaxPooling2D(name="pool2", pool_size=(2, 2), border_mode='same')(conv2)

    conv3 = Convolution2D(name="deconv1", nb_filter=32, nb_row=3, nb_col=3, activation='relu', border_mode='same')(pool2)
    upsamp1 = UpSampling2D(name="up1", size=(2, 2))(conv3)
    conv4 = Convolution2D(name="deconv2", nb_filter=32, nb_row=3, nb_col=3, activation='relu', border_mode='same')(upsamp1)
    upsamp2 = UpSampling2D(name="up2", size=(2, 2))(conv4)

    output_img = Convolution2D(name="output", nb_filter=num_channels, nb_row=3, nb_col=3, activation='sigmoid', border_mode='same')(upsamp2)

    model = Model(input_img, output_img)

    return model

def FI_CNN_model_BN(num_channels):
    input_img = Input(name="input", shape=(num_channels*2, 36, 64))

    conv1 = Convolution2D(name="conv1", nb_filter=32, nb_row=3, nb_col=3, activation='relu', border_mode='same')(input_img)
    bn1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(name="pool1", pool_size=(2, 2), border_mode='same')(bn1)
    conv2 = Convolution2D(name="conv2", nb_filter=32, nb_row=3, nb_col=3, activation='relu', border_mode='same')(pool1)
    bn2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(name="pool2", pool_size=(2, 2), border_mode='same')(bn2)

    conv3 = Convolution2D(name="deconv1", nb_filter=32, nb_row=3, nb_col=3, activation='relu', border_mode='same')(pool2)
    bn3 = BatchNormalization()(conv3)
    upsamp1 = UpSampling2D(name="up1", size=(2, 2))(bn3)
    conv4 = Convolution2D(name="deconv2", nb_filter=32, nb_row=3, nb_col=3, activation='relu', border_mode='same')(upsamp1)
    bn4 = BatchNormalization()(conv4)
    upsamp2 = UpSampling2D(name="up2", size=(2, 2))(bn4)

    output_img = Convolution2D(name="output", nb_filter=num_channels, nb_row=3, nb_col=3, activation='sigmoid', border_mode='same')(upsamp2)

    model = Model(input_img, output_img)

    return model