# This script takes in an [x] FPS video and outputs 2*[x] FPS video by applying results
#  of frame interpolation using FI-CNN.

import sys
import os

# BACKEND = "theano"
BACKEND = "tensorflow"

os.environ['KERAS_BACKEND'] = BACKEND
os.environ['THEANO_FLAGS'] = "device=gpu0, lib.cnmem=0.85, optimizer=fast_run"

import sys
import os
import time
import random

import numpy as np
from scipy.misc import imread, imsave, imshow, imresize
import cv2

from FI_unet import get_unet_2


def load_vid(vid_path):
    cap = cv2.VideoCapture(vid_path)

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    vid_arr = np.zeros(shape=(num_frames, 128, 384, 3), dtype="uint8")

    for i in range(num_frames):
        if i % (num_frames / 10) == 0:
            print ("Video loading is {0}% done.".format((i / (num_frames / 10) * 10)))

        ret, frame = cap.read()

        vid_arr[i] = imresize(frame, (128, 384))

    return vid_arr, fps

def double_vid_fps(vid_arr):

    model = get_unet_2((6, 128, 384))
    model.load_weights("./../model_weights/weights_unet2_finetune_youtube_100epochs.hdf5")

    # new_vid_arr = np.zeros(shape=(len(vid_arr)*2, 128, 384, 3))
    new_vid_arr = []
    new_vid_arr.append(vid_arr[0])
    for i in range(1, len(vid_arr)):
        if i % (len(vid_arr) / 10) == 0:
            print ("FPS doubling is {0}% done.".format((i / (len(vid_arr) / 10) * 10)))


        pred = model.predict(np.expand_dims(np.transpose(np.concatenate((vid_arr[i-1], vid_arr[i]), axis=2)/255., (2, 0, 1)), axis=0))
        new_vid_arr.append((np.transpose(pred[0], (1, 2, 0))*255).astype("uint8"))
        new_vid_arr.append(vid_arr[i])

    return np.asarray(new_vid_arr)

def save_vid(vid_arr, vid_out_path, fps):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # fourcc = cv2.VideoWriter_fourcc(*'X264')
    out = cv2.VideoWriter(vid_out_path, fourcc, fps, (384, 128))

    for i in range(len(vid_arr)):
        out.write(vid_arr[i])

def main():

    vid_dir = "/media/neil/Neil's 240GB SSD/deep-motion_data/"
    # vid_fn = "30sec_nature.mp4"
    vid_fn = "planet_earth_2.mp4"
    out_dir = "./../results/videos/"

    vid_arr, fps = load_vid(os.path.join(vid_dir, vid_fn))

    double_vid_arr = double_vid_fps(vid_arr)

    save_vid(vid_arr, out_dir + vid_fn.split('.')[0] + "_resize.avi", fps=fps)
    save_vid(double_vid_arr, out_dir + vid_fn.split('.')[0] + "_double_60.avi", fps=fps*2)
    save_vid(double_vid_arr, out_dir + vid_fn.split('.')[0] + "_double_30.avi", fps=fps)

    quad_vid_arr = double_vid_fps(double_vid_arr)
    save_vid(quad_vid_arr, out_dir + vid_fn.split('.')[0] + "_quad_30.avi", fps=fps)

if __name__ == '__main__':
    main()

