import sys
import os
import time
import random
import subprocess as sp
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread, imsave, imshow, imresize, imsave
from skimage import color

from multiprocessing import Pool, cpu_count
from functools import partial

FFMPEG_BIN = "ffmpeg"
FFPROBE_BIN = "ffprobe"

# VID_DIR = "/media/neil/Neil's 5TB HDD/deep-motion_data/youtube-8m-videos"
VID_DIR = "/media/neil/Neil's 240GB SSD/deep-motion_data/youtube-8m-videos"

KITTI_VID_DIR = "/media/neil/Neil's 240GB SSD/deep-motion_data/KITTI_RAW/train/"

FRAME_DISTS = [3, 5, 7, 9]


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

    # im = zoom(im, min_zoom=final_im_size, max_zoom=1.0, rand_seed=rand_seed)
    im = imresize(im, final_im_size)
    im = crop(im, crop_size=final_im_size, crop_corner_loc="random", random_crop_amount=1.0, rand_seed=rand_seed)

    im = np.transpose(im, (2, 0, 1))

    return im


def batch_generator(batch_size, num_channels, batch_image_size):
    vid_list = os.listdir(VID_DIR)

    while 1:
        if batch_image_size == "random":
            batch_image_size_mult = random.randint(4, 80)
            batch_image_size = (9*batch_image_size_mult, 16*batch_image_size_mult)

        first_frame_batch = np.zeros(shape=(batch_size, 720, 1280, 3), dtype="uint8")
        middle_frame_batch = np.zeros(shape=(batch_size, 720, 1280, 3), dtype="uint8")
        last_frame_batch = np.zeros(shape=(batch_size, 720, 1280, 3), dtype="uint8")

        random.seed()
        frame_dist = random.choice(FRAME_DISTS)
        i = 0
        while i < batch_size:
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

            dif = np.mean(np.abs((last_frame_batch[i] / 255.) - (first_frame_batch[i] / 255.)))
            if dif < 0.02 or dif > 0.2:
                continue

            # print dif
            # X_blend = (first_frame_batch[i]/255. + last_frame_batch[i]/255.) * 255. / 2
            # plt.figure()
            # plt.title("First Frame")
            # plt.imshow(X_blend.astype("uint8"))

            i += 1

        rand_seeds = random.sample(range(0, 2048), batch_size) * 3
        batch_before_transform = zip(list(np.concatenate((first_frame_batch, last_frame_batch, middle_frame_batch))), rand_seeds)
        batch_after_transform = transform_batch_parallel(batch_before_transform, num_channels=num_channels, final_im_size=batch_image_size)
        # batch_after_transform = transform_batch(batch_before_transform, num_channels=NUM_CHANNELS, final_im_size=batch_im_size)

        X_batch = np.concatenate((batch_after_transform[:batch_size], batch_after_transform[batch_size:batch_size*2]), axis=1)
        y_batch = batch_after_transform[batch_size*2:]

        yield X_batch.astype("float32") / 255., y_batch.astype("float32") / 255.


def kitti_batch_generator(batch_size, frame_dists=(2, 4, 6), data_aug=True):
    raw_im_list = open(os.path.join(KITTI_VID_DIR, "im_list.txt")).read().splitlines()
    im_list = []
    for im in raw_im_list:
        if not int(im.split('/')[-1].split('.')[0]) < max(frame_dists):
            im_list.append(im)

    while 1:
        X = np.zeros(shape=(batch_size, 128, 384, 6), dtype="uint8")
        y = np.zeros(shape=(batch_size, 128, 384, 3), dtype="uint8")
        for batch_i in range(batch_size):
            random.seed()
            im_path = random.choice(im_list)
            frame_dist = random.choice(frame_dists)
            frame_num = int(im_path.split('/')[-1].split('.')[0])
            X[batch_i, :, :, :3] = imread(os.path.join(KITTI_VID_DIR, '/'.join(im_path.split('/')[:-1]), str(frame_num-frame_dist).zfill(10) + ".png"))[:, 20:404, :]
            X[batch_i, :, :, 3:] = imread(os.path.join(KITTI_VID_DIR, '/'.join(im_path.split('/')[:-1]), str(frame_num).zfill(10) + ".png"))[:, 20:404, :]
            y[batch_i] = imread(os.path.join(KITTI_VID_DIR, '/'.join(im_path.split('/')[:-1]), str(frame_num-(frame_dist/2)).zfill(10) + ".png"))[:, 20:404, :]

            if data_aug:
                if random.randint(0, 1):
                    X[batch_i, :, :, :3] = np.fliplr(X[batch_i, :, :, :3])
                    X[batch_i, :, :, 3:] = np.fliplr(X[batch_i, :, :, 3:])
                    y[batch_i] = np.fliplr(y[batch_i])

                if random.randint(0, 1):
                    X[batch_i, :, :, :3] = np.flipud(X[batch_i, :, :, :3])
                    X[batch_i, :, :, 3:] = np.flipud(X[batch_i, :, :, 3:])
                    y[batch_i] = np.flipud(y[batch_i])

        yield np.transpose(X, (0, 3, 1, 2)).astype("float32") / 255., np.transpose(y, (0, 3, 1, 2)).astype("float32") / 255.


def main():
    # X, y = kitti_batch_generator(500).next()
    # np.save("X_val_KITTI.npy", X)
    # np.save("y_val_KITTI.npy", y)


    # total = 95403
    # count = 0
    # im_list = []
    # for path, subdirs, files in os.walk(KITTI_VID_DIR):
    #     for name in files:
    #         fn = os.path.join(path, name)
    #         if fn.endswith(".png"):
    #             im_list.append(fn[len(KITTI_VID_DIR):])
    #             # im = imread(fn)
    #             # print im.shape
    #             # if not (im.shape[1] * 128.0) / im.shape[0] >= 423 and (im.shape[1] * 128.0) / im.shape[0] <= 425:
    #             # if im.shape != (128, 424, 3):
    #             #     print "FAIL", fn
    #                 # continue
    #             # imsave(fn, imresize(im, (128, 424)))
    #             # print count
    #             count += 1
    # print '\n'.join(im_list)
    # exit()

    # raw_input("go")
    # core = 6
    # im_i = 1
    # # for im_fn in im_list[core*11926:(core+1)*11926]:
    # for im_fn in im_list[core*11926+5000:(core+1)*11926]:
    #     im = imread(im_fn)
    #     if im.shape != (128, 424, 3):
    #         print "HERE"
    #         imsave(im_fn, imresize(im, (128, 424)))
    #     print im_i, "images done"
    #     im_i += 1

    # files = os.listdir(".")
    #
    # X = np.zeros(shape=(120000, 6, 36, 64), dtype="uint8")
    # y = np.zeros(shape=(120000, 3, 36, 64), dtype="uint8")
    # j = 0
    # for i in range(len(files)):
    #     if files[i].startswith("X_CORE"):
    #
    #         X[j*100:(j+1)*100] = np.load(files[i]).astype("uint8")
    #         y[j*100:(j+1)*100] = np.load("y"+files[i][1:]).astype("uint8")
    #         j += 1
    #
    # from sklearn.utils import shuffle
    # (X, y) = shuffle(X, y)
    #
    # np.save("X_small_train", X[:100000])
    # np.save("y_small_train", y[:100000])
    # np.save("X_small_val", X[100000:110000])
    # np.save("y_small_val", y[100000:110000])
    # np.save("X_small_test", X[110000:120000])
    # np.save("y_small_test", y[110000:120000])

    start_time = time.time()
    i = 0
    batch_start_time = time.time()
    BATCH_SIZE = 100
    MAX_BATCHES = 100000 / BATCH_SIZE

    gen = kitti_batch_generator(200)
    # gen = batch_generator(batch_size=BATCH_SIZE, num_channels=3, batch_image_size=(36, 64))
    # for X, y in batch_generator(batch_size=BATCH_SIZE, num_channels=3, batch_image_size = "random"):
    while i < MAX_BATCHES:
        try:
            X, y = gen.next()
        except:
            print "ERROR: BATCH GEN FAILED, skipping..."
            continue
        print "Time for batch:", time.time() - batch_start_time, "seconds"

        print X.shape
        print y.shape
        print i

        # np.save("X_CORE1_small_train_" + str(i), X)
        # np.save("y_CORE1_small_train_" + str(i), y)

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

        # if i % 10 == 0:
        #     print "myBatchGenerator:", i, "batches done in", (time.time() - start_time) / 60.0, "minutes..."
        # if i == 100:
        #     break
        i += 1

        batch_start_time = time.time()
    print time.time() - start_time



if __name__ == '__main__':
    main()
