import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import cpu_count

from keras.optimizers import SGD, adadelta, adagrad, adam, adamax, nadam
from keras.callbacks import ModelCheckpoint, TensorBoard

from FI_CNN import FI_CNN_model
from data_generator import batch_generator

BATCH_SIZE = 20
NUM_EPOCHS = 1000
NUM_CHANNELS = 3

DO_TESTING = False

def main():
    X_train = np.load("X_train.npy")[:500].astype("float32") / 255.
    y_train = np.load("y_train.npy")[:500].astype("float32") / 255.
    X_val = np.load("X_val.npy")[:100].astype("float32") / 255.
    y_val = np.load("y_val.npy")[:100].astype("float32") / 255.


    model = FI_CNN_model(NUM_CHANNELS)

    # optimizer = SGD(lr= 0.0001, decay = 0., momentum = 0.9, nesterov = True)
    optimizer = adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    # loss = "mse"
    loss = "categorical_crossentropy"
    # loss = "binary_crossentropy"

    model.compile(loss=loss, optimizer=optimizer)

    callbacks = [
        ModelCheckpoint(filepath="./../model_weights/weights.hdf5", monitor='val_loss',
                        save_best_only=True, verbose=1),
    ]
    callbacks.append(TensorBoard(log_dir="./../tensorboard_logs", histogram_freq=1))

    if DO_TESTING:
        model.load_weights("./../model_weights/weights.hdf5")
        X, y = batch_generator(BATCH_SIZE, NUM_CHANNELS, batch_image_size="random").next()
        y_pred = model.predict_on_batch(X)
        # # code to inspect images in batch
        for i in range(len(X)):
            X_0 = X[i, :3, :, :]
            X_1 = X[i, 3:, :, :]
            plt.figure()
            plt.title("First Frame")
            plt.imshow(np.transpose(X_0, (1, 2, 0)).astype("uint8"))
            plt.figure()
            plt.title("Middle Frame")
            plt.imshow(np.transpose(y[i], (1, 2, 0)).astype("uint8"))
            plt.figure()
            plt.title("Last Frame")
            plt.imshow(np.transpose(X_1, (1, 2, 0)).astype("uint8"))
            plt.figure()
            plt.title("Predicted Middle Frame")
            plt.imshow(np.transpose(y_pred[i], (1, 2, 0)).astype("uint8"))


    # BATCH_IMAGE_SIZE = "random"
    BATCH_IMAGE_SIZE = (36, 64)
    print "Begin training..."
    # hist = model.fit_generator(
    #     generator=batch_generator(BATCH_SIZE, NUM_CHANNELS, BATCH_IMAGE_SIZE),
    #     samples_per_epoch=BATCH_SIZE*10,
    #     nb_epoch=NUM_EPOCHS,
    #     callbacks=callbacks,
    #     max_q_size=10,
    #     nb_worker=1,
    #     # nb_worker=cpu_count(),
    # )

    hist = model.fit(x=X_train, y=y_train, batch_size=BATCH_SIZE, nb_epoch=NUM_EPOCHS, callbacks=callbacks, validation_data=(X_val, y_val), shuffle=True)
if __name__ == '__main__':
    main()
