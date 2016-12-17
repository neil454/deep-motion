import sys
import os
import time
import random
import subprocess as sp
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread, imsave, imshow, imresize
from skimage import color

from multiprocessing import Pool, cpu_count
from functools import partial

FFMPEG_BIN = "ffmpeg"
FFPROBE_BIN = "ffprobe"

# VID_DIR = "/media/neil/Neil's 5TB HDD/deep-motion_data/youtube-8m-videos"
VID_DIR = "/media/neil/Neil's 240GB SSD/deep-motion_data/youtube-8m-videos"

FRAME_DISTS = [3, 5, 7, 9]

# random.seed()
# random.shuffle(files)
#
# frames = np.zeros(shape=(50, 3, 720, 1280, 3))
#
# start_time = time.time()
# for i in range(len(files[:50])):
#     # cap = cv2.VideoCapture(os.path.join(VID_DIR, files[i]))
#
#     # frame_num = np.random.randint(0, cap.get(cv2.CAP_PROP_FRAME_COUNT)
#     # num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
#     frame_num = 200
#
#     command = [FFPROBE_BIN, '-show_format', '-loglevel', 'warning', os.path.join(VID_DIR, files[i])]
#     pipe = sp.Popen(command, stdout=sp.PIPE)
#     pipe.stdout.readline()
#     pipe.terminate()
#     infos = pipe.stdout.read()
#     duration_index = infos.find("duration=") + 9
#     duration_length = infos[duration_index:].find("\nsize=")
#     duration = float(infos[duration_index:duration_index+duration_length])
#     # rand_time = random.uniform(0.0, duration-1)
#     command = [FFMPEG_BIN,
#                '-ss', '2.123456',
#                # '-ss', str(rand_time),
#                '-i', os.path.join(VID_DIR, files[i]),
#                '-frames:v', '3',
#                '-f', 'image2pipe',
#                '-pix_fmt', 'rgb24',
#                '-loglevel', 'warning',
#                '-vcodec', 'rawvideo', '-']
#     pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=10 ** 8)
#     # read 1280*720*3 bytes (= 1 frame)
#     raw_image = pipe.stdout.read(1280 * 720 * 3 * 3)
#
#     # transform the byte read into a numpy array
#     frames[i, :] = np.fromstring(raw_image, dtype='uint8').reshape((3, 720, 1280, 3))
#     # throw away the data in the pipe's buffer.
#     pipe.stdout.flush()
#
#     # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
#     # ret, frame = cap.read()
#     # frames[i, 0] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num+1)
#     # ret, frame = cap.read()
#     # frames[i, 1] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num+2)
#     # ret, frame = cap.read()
#     # frames[i, 2] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#
# print "Done in", time.time() - start_time, "seconds."
# a = 0

# min/max_zoom can be either:
#  - tuple of ints for zoomed image size (must be proportional to original image size,
#  - float for zoom factor
def zoom(im, min_zoom, max_zoom, rand_seed=None):
    np.random.seed(rand_seed)

    # calculate min/max_zoom factor if they are tuples of ints (for image size)
    if isinstance(min_zoom, tuple):
        min_zoom = max(float(min_zoom[0])/im.shape[0], float(min_zoom[1])/im.shape[1])
    if isinstance(max_zoom, tuple):
        max_zoom = max(float(max_zoom[0])/im.shape[0], float(max_zoom[1])/im.shape[1])

    if min_zoom == max_zoom:
        zoom_factor = min_zoom
    else:
        zoom_factor = np.random.uniform(low=min_zoom, high=max_zoom)

    # im = scipy.ndimage.zoom(im, zoom_factor)
    im = imresize(im, zoom_factor)

    return im

def crop(im, crop_size, crop_corner_loc="center", random_crop_amount=1.0, rand_seed=None):
    np.random.seed(rand_seed)

    # Handle cases of special crop_corner_loc values
    if isinstance(crop_corner_loc, tuple):
        crop_corner_loc = crop_corner_loc
    elif crop_corner_loc == "random":
        # set crop_corner_loc to crop center
        crop_corner_loc = ((im.shape[0] / 2) - (crop_size[0] / 2), (im.shape[1] / 2) - (crop_size[1] / 2))
        # if random_crop_amount not a tuple, make it one, and use it's value for both row and col amounts
        if not isinstance(random_crop_amount, tuple):
            random_crop_amount = (random_crop_amount, random_crop_amount)

        # print "im", im.shape
        # print "crop_corner_loc", crop_corner_loc

        # calculate the allowable shift of the crop, based on random_crop_amount percentage
        crop_allowable_shift = (int(crop_corner_loc[0] * random_crop_amount[0]),
                                int(crop_corner_loc[1] * random_crop_amount[1]))
        # print "crop_allowable_shift", crop_allowable_shift


        # shift crop randomly in range 0 to crop_allowable_shift, in either positive or negative direction
        # need to convert to list temp to do assignment...
        crop_corner_loc = list(crop_corner_loc)
        if crop_allowable_shift[0] != 0:
            crop_corner_loc[0] = crop_corner_loc[0] + np.random.randint(-1 * crop_allowable_shift[0], crop_allowable_shift[0])
        if crop_allowable_shift[1] != 0:
            crop_corner_loc[1] = crop_corner_loc[1] + np.random.randint(-1 * crop_allowable_shift[1], crop_allowable_shift[1])
        crop_corner_loc = tuple(crop_corner_loc)

    else:
        # set crop_corner_loc to crop center
        crop_corner_loc = ((im.shape[0] / 2) - (crop_size[0] / 2), (im.shape[1] / 2) - (crop_size[1] / 2))

    # crop image
    im = im[crop_corner_loc[0]: crop_corner_loc[0] + crop_size[0],
            crop_corner_loc[1]: crop_corner_loc[1] + crop_size[1],
            :]

    return im

# Takes in a batch of images, and augments them (normal version, doesn't use multiprocessing. Better when not image aug is disabled)
def transform_batch(batch, num_channels, final_im_size):
    batch_trans = np.zeros(shape=(len(batch), num_channels) + final_im_size)
    for i in range(len(batch)):
        batch_trans[i] = transform_im(num_channels, final_im_size, batch[i])

    # # Debug code: Compare image before/after augmentation
    # for i in range(len(X)):
    #     show_image(i, X)
    #     show_image(i, X_aug)

    return batch_trans


# Takes in a batch of images, and augments them (uses multiprocessing to augment images in parallel)
def transform_batch_parallel(batch, num_channels, final_im_size):
    p = Pool(cpu_count())
    augment_im_partial = partial(transform_im, num_channels, final_im_size)
    batch_trans = p.map(augment_im_partial, batch)
    p.close()
    p.join()
    batch_trans = np.asarray(batch_trans, dtype="float32")

    # # Debug code: Compare image before/after augmentation
    # for i in range(len(X)):
    #     show_image(i, X)
    #     show_image(i, X_aug)

    return batch_trans

def transform_im(num_channels, final_im_size, batch_i):
    im = batch_i[0]
    rand_seed = batch_i[1]

    if num_channels == 1:
        im = color.rgb2gray(im)

    im = zoom(im, min_zoom=final_im_size, max_zoom=1.0, rand_seed=rand_seed)
    im = crop(im, crop_size=final_im_size, crop_corner_loc="random", random_crop_amount=1.0, rand_seed=rand_seed)

    im = np.transpose(im, (2, 0, 1))

    return im

def batch_generator(batch_size, num_channels, batch_image_size):
    vid_list = os.listdir(VID_DIR)

    while 1:
        if batch_image_size == "random":
            batch_im_size_mult = random.randint(4, 80)
            batch_im_size = (9*batch_im_size_mult, 16*batch_im_size_mult)

        first_frame_batch = np.zeros(shape=(batch_size, 720, 1280, 3), dtype="uint8")
        middle_frame_batch = np.zeros(shape=(batch_size, 720, 1280, 3), dtype="uint8")
        last_frame_batch = np.zeros(shape=(batch_size, 720, 1280, 3), dtype="uint8")

        random.seed()
        frame_dist = random.choice(FRAME_DISTS)
        for i in range(batch_size):
            vid_path = os.path.join(VID_DIR, random.choice(vid_list))
            command = [FFPROBE_BIN, '-show_format', '-loglevel', 'warning', vid_path]
            pipe = sp.Popen(command, stdout=sp.PIPE)
            pipe.stdout.readline()
            pipe.terminate()
            infos = pipe.stdout.read()
            duration_index = infos.find("duration=") + 9
            duration_length = infos[duration_index:].find("\nsize=")
            duration = float(infos[duration_index:duration_index + duration_length])
            rand_time = random.uniform(0.0, duration-1)
            command = [FFMPEG_BIN,
                       '-ss', str(rand_time),
                       '-i', vid_path,
                       '-frames:v', str(frame_dist),
                       '-f', 'image2pipe',
                       '-pix_fmt', 'rgb24',
                       '-loglevel', 'warning',
                       '-vcodec', 'rawvideo', '-']
            pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=10 ** 8)
            raw_image = pipe.stdout.read(frame_dist * 720 * 1280 * 3)

            # transform the byte read into a numpy array
            frames = np.fromstring(raw_image, dtype='uint8').reshape((frame_dist, 720, 1280, 3))

            first_frame_batch[i] = frames[0]
            middle_frame_batch[i] = frames[len(frames)/2]
            last_frame_batch[i] = frames[-1]

            # throw away the data in the pipe's buffer.
            pipe.stdout.flush()

        rand_seeds = random.sample(range(0, 2048), batch_size) * 3
        batch_before_transform = zip(list(np.concatenate((first_frame_batch, last_frame_batch, middle_frame_batch))), rand_seeds)
        batch_after_transform = transform_batch_parallel(batch_before_transform, num_channels=num_channels, final_im_size=batch_im_size)
        # batch_after_transform = transform_batch(batch_before_transform, num_channels=NUM_CHANNELS, final_im_size=batch_im_size)

        X_batch = np.concatenate((batch_after_transform[:batch_size], batch_after_transform[batch_size:batch_size*2]), axis=1)
        y_batch = batch_after_transform[batch_size*2:]

        yield X_batch, y_batch

def main():
    start_time = time.time()
    i = 0
    batch_start_time = time.time()
    BATCH_SIZE = 100
    MAX_BATCHES = 100000 / BATCH_SIZE

    gen = batch_generator(batch_size=BATCH_SIZE, num_channels=3, batch_image_size=(36, 64))
    # for X, y in batch_generator(batch_size=BATCH_SIZE, num_channels=3, batch_image_size = "random"):
    while i < MAX_BATCHES:
        try:
            X, y = gen.next()
        except:
            continue
        print "Time for batch:", time.time() - batch_start_time, "seconds"

        print X.shape
        print y.shape
        print i

        np.save("X_small_train_" + str(i), X)
        np.save("y_small_train_" + str(i), y)

        # # # code to inspect images in batch
        # for i in range(len(X)):
        #     X_0 = X[i, :3, :, :]
        #     X_1 = X[i, 3:, :, :]
        #     plt.figure()
        #     plt.title("First Frame")
        #     plt.imshow(np.transpose(X_0, (1, 2, 0)).astype("uint8"))
        #     plt.figure()
        #     plt.title("Middle Frame")
        #     plt.imshow(np.transpose(y[i], (1, 2, 0)).astype("uint8"))
        #     plt.figure()
        #     plt.title("Last Frame")
        #     plt.imshow(np.transpose(X_1, (1, 2, 0)).astype("uint8"))

        # if i % 10 == 0:
        #     print "myBatchGenerator:", i, "batches done in", (time.time() - start_time) / 60.0, "minutes..."
        # if i == 100:
        #     break
        i += 1

        batch_start_time = time.time()
    print time.time() - start_time



if __name__ == '__main__':
    main()
