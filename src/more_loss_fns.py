# This file contains my custom loss functions I experimented with
# look at charbonnier() to see the loss function I decided to go with

import numpy as np
from keras import objectives
from keras import backend as K

_EPSILON = K.epsilon()

def charbonnier(y_true, y_pred):
    return K.sqrt(K.square(y_true - y_pred) + 0.01**2)

def huber(y_true, y_pred):
    d = y_true - y_pred
    a = .5 * K.square(d)
    b = 0.1 * (K.abs(d) - 0.1 / 2.)
    l = K.switch(K.abs(d) <= 0.1, a, b)
    return K.sum(l, axis=-1)


def smooth_huber(y_true, y_pred):
    return K.mean(K.log(np.cosh(y_true - y_pred)))

def binary_crossentropy_test(y_true, y_pred):
    y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
    out = -(y_true * K.log(y_pred) + (1.0 - y_true) * K.log(1.0 - y_pred))
    return K.mean(out, axis=-1)