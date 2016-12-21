import sys
import os

# BACKEND = "theano"
BACKEND = "tensorflow"

os.environ['KERAS_BACKEND'] = BACKEND
os.environ['THEANO_FLAGS'] = "device=gpu0, lib.cnmem=0.85, optimizer=fast_run"

import random
import logging
import numpy as np
from scipy.misc import imsave
import matplotlib.pyplot as plt
from multiprocessing import cpu_count

from keras.optimizers import SGD, adadelta, adagrad, adam, adamax, nadam
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, ReduceLROnPlateau

from FI_CNN import FI_CNN_model, FI_CNN_model_BN
from FI_unet import *
from data_generator import batch_generator, kitti_batch_generator

from more_loss_fns import *

LEARNING_RATE = 0.0001
BATCH_SIZE = 16
NUM_EPOCHS = 1000
NUM_CHANNELS = 3

LOAD_PRE_TRAINED_MODEL = True
DO_TESTING = True


# need to use this when I require batches from an already memory-loaded X, y
def np_array_batch_generator(X, y, batch_size):
    batch_i = 0
    while 1:
        if (batch_i+1)*batch_size >= len(X):
            yield X[batch_i*batch_size:], y[batch_i*batch_size:]
            batch_i = 0
        else:
            yield X[batch_i*batch_size:(batch_i+1)*batch_size], y[batch_i*batch_size:(batch_i+1)*batch_size]


def main():
    ##### DATA SETUP #####
    # X_train = np.load("X_small_train.npy")[:500, :, :32].astype("float32") / 255.
    # y_train = np.load("y_small_train.npy")[:500, :, :32].astype("float32") / 255.
    # X_val = np.load("X_small_val.npy")[:1000, :, :32].astype("float32") / 255.
    # y_val = np.load("y_small_val.npy")[:1000, :, :32].astype("float32") / 255.

    # X_train = np.load("X_train.npy")[1050:, :, :320].astype("float32") / 255.
    # y_train = np.load("y_train.npy")[1050:, :, :320].astype("float32") / 255.
    # X_train = np.load("X_train.npy")[50:1050, :, :320].astype("float32") / 255.
    # y_train = np.load("y_train.npy")[50:1050, :, :320].astype("float32") / 255.
    # X_val = np.load("X_val.npy")[:100, :, :320].astype("float32") / 255.
    # y_val = np.load("y_val.npy")[:100, :, :320].astype("float32") / 255.

    X_val = np.load("X_val_KITTI.npy").astype("float32") / 255.
    y_val = np.load("y_val_KITTI.npy").astype("float32") / 255.

    ##### MODEL SETUP #####
    # model = FI_CNN_model(NUM_CHANNELS)
    # model = FI_CNN_model_BN(NUM_CHANNELS)
    # model = get_unet()
    model = get_unet_2(input_shape=(6, 128, 384))
    # model = get_unet_3(input_shape=(6, 128, 384), batch_size=BATCH_SIZE)

    # optimizer = SGD(lr=LEARNING_RATE, decay = 0., momentum = 0.9, nesterov = True)
    optimizer = adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    # loss = "hinge"
    # loss = "mse"
    # loss = "categorical_crossentropy"
    # loss = "binary_crossentropy"
    loss = charbonnier

    model.compile(loss=loss, optimizer=optimizer)


    if DO_TESTING:
        model.load_weights("./../model_weights/weights_kitti_167plus25epochs_unet2_ch_pt03136_best_but_fixed_imsize.hdf5")
        # X, y = batch_generator(50, NUM_CHANNELS, batch_image_size=(128, 384)).next()
        X, y = kitti_batch_generator(50).next()
        # X = X_val
        # y = y_val
        # X = X_train
        # y = y_train
        y_pred = model.predict(X, batch_size=BATCH_SIZE, verbose=1)

        model.load_weights("./../model_weights/weights_unet2_finetune_youtube_100epochs.hdf5")

        y_pred_2 = model.predict(X, batch_size=BATCH_SIZE, verbose=1)
        # # code to inspect images in batch
        for i in range(len(X)):
            X_0 = X[i, :3, :, :]
            X_1 = X[i, 3:, :, :]
            X_blend = (X_0 + X_1) * 255. / 2
            plt.figure()
            plt.title("First Frame")
            plt.imshow((np.transpose(X_0, (1, 2, 0))*255).astype("uint8"))
            plt.figure()
            plt.title("Middle Frame")
            plt.imshow((np.transpose(y[i], (1, 2, 0))*255).astype("uint8"))
            plt.figure()
            plt.title("Last Frame")
            plt.imshow((np.transpose(X_1, (1, 2, 0))*255).astype("uint8"))
            plt.figure()
            plt.title("Blended")
            plt.imshow(np.transpose(X_blend, (1, 2, 0)).astype("uint8"))
            plt.figure()
            plt.title("Predicted Middle Frame")
            plt.imshow((np.transpose(y_pred[i], (1, 2, 0))*255).astype("uint8"))
            plt.figure()
            plt.title("Predicted Middle Frame (After Finetuning)")
            plt.imshow((np.transpose(y_pred_2[i], (1, 2, 0)) * 255).astype("uint8"))

            save = False
            if save:
                id = random.randint(0, 100000)
                imsave("./../results/preds/" + str(i) + "_First_Frame_" + str(id) + ".png", (np.transpose(X_0, (1, 2, 0))*255).astype("uint8"))
                imsave("./../results/preds/" + str(i) + "_Middle_Frame_" + str(id) + ".png", (np.transpose(y[i], (1, 2, 0))*255).astype("uint8"))
                imsave("./../results/preds/" + str(i) + "_Last_Frame_" + str(id) + ".png", (np.transpose(X_1, (1, 2, 0))*255).astype("uint8"))
                imsave("./../results/preds/" + str(i) + "_Blended_" + str(id) + ".png", (np.transpose(X_blend, (1, 2, 0)).astype("uint8")))
                imsave("./../results/preds/" + str(i) + "_Predicted_Middle_Frame_" + str(id) + ".png", (np.transpose(y_pred[i], (1, 2, 0))*255).astype("uint8"))
                imsave("./../results/preds/" + str(i) + "_Predicted_Middle_Frame_FT_" + str(id) + ".png", (np.transpose(y_pred_2[i], (1, 2, 0))*255).astype("uint8"))
        exit()

    ##### TRAINING SETUP #####
    logging.warning("USING loss AS MODEL CHECKPOINT METRIC, CHANGE LATER!")
    callbacks = [
        # ModelCheckpoint(filepath="./../model_weights/weights.hdf5", monitor='val_loss', save_best_only=True, verbose=1),
        ModelCheckpoint(filepath="./../model_weights/weights.hdf5", monitor='loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="loss", factor=0.5, patience=20, verbose=1)
    ]
    # callbacks.append(TensorBoard(log_dir="./../tensorboard_logs", write_graph=False))

    if LOAD_PRE_TRAINED_MODEL:
        print ""
        logging.warning("LOADING PRE-TRAINED MODEL WEIGHTS!")
        print ""
        model.load_weights("./../weights.hdf5")
        callbacks.append(CSVLogger("stats_per_epoch.csv", append=True))
    else:
        callbacks.append(CSVLogger("stats_per_epoch.csv", append=False))


    # OPTION 1: train on batches from youtube-8m
    # BATCH_IMAGE_SIZE = "random"
    # BATCH_IMAGE_SIZE = (320, 640)
    BATCH_IMAGE_SIZE = (128, 384)
    print "Begin training..."
    hist = model.fit_generator(
        generator=batch_generator(BATCH_SIZE, NUM_CHANNELS, BATCH_IMAGE_SIZE),
        samples_per_epoch=800,
        nb_epoch=NUM_EPOCHS,
        callbacks=callbacks,
        validation_data=(X_val, y_val),
        max_q_size=10,
        nb_worker=1,
        # nb_worker=cpu_count(),
    )

    # OPTION 2: train on batches from kitti
    # hist = model.fit_generator(
    #     generator=kitti_batch_generator(BATCH_SIZE),
    #     samples_per_epoch=800,
    #     nb_epoch=NUM_EPOCHS,
    #     callbacks=callbacks,
    #     validation_data=np_array_batch_generator(X_val, y_val, BATCH_SIZE),
    #     nb_val_samples=len(X_val),
    #     max_q_size=10,
    #     nb_worker=1,
    #     # nb_worker=cpu_count(),    # deprecated
    # )

    # OPTION 3: train on some memory-loaded data
    # hist = model.fit(x=X_train, y=y_train, batch_size=BATCH_SIZE, nb_epoch=NUM_EPOCHS, callbacks=callbacks, validation_data=(X_val, y_val), shuffle=True)
if __name__ == '__main__':
    main()
