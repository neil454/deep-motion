# Deep Motion: A Convolutional Neural Network for Frame Interpolation
Use a DCNN to perform frame interpolation.

Paper: [Deep Motion: A Convolutional Neural Network for Frame Interpolation](https://github.com/neil454/deep-motion/raw/master/deep-motion_paper.pdf)

Based off of the [U-Net](https://arxiv.org/abs/1505.04597) architecture

<img src="http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png" alt="alt text" width="750" height="500">


## Software Requirements
 - Keras (tested on v1.1.2)
 - TensorFlow (tested on v0.10.0)
 - NumPy, SciPy, matplotlib
 - OpenCV (tested on v3.1.0, but v2.X should work) (only needed for fps_convert.py)
 - FFMPEG (only needed for batch samples generator for YouTube-8M videos)


## Model Weights
Download the model weights [here](https://github.com/neil454/deep-motion/releases/download/0.1/weights_unet2_finetune_youtube_100epochs.hdf5).

*Note that the weights are trained using the architecture defined in `FI_unet.py/get_unet_2()`, which requires input of `shape=(6, 128, 384)`, due to the use of Batch Normalization (probably could do without that)


## Training
Details in `train.py`. It's Keras, so don't worry ;)


## Testing
For images, look at `DO_TESTING` section of `train.py`

For videos, you can use `fps_convert.py` to double/quadruple/etc the FPS of any video


## Results
![](https://raw.githubusercontent.com/neil454/deep-motion/master/results/planet_earth_interpolation_results.gif)

View the results at the end of the [paper](https://github.com/neil454/deep-motion/raw/master/deep-motion_paper.pdf)

Watch the [presentation video](https://www.youtube.com/watch?v=RWaWoQWI4ks)

[Presentation Slides](https://github.com/neil454/deep-motion/raw/master/deep-motion_slides.pdf)
