# Tiny ImageNet

## Introduction

I doubt that is controversial to say that [ImageNet](http://www.image-net.org/) and Alex Krizhevsky's ["AlexNet"](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) sparked a revolution in machine learning. By and large, AlexNet marked the end of the era of hand-crafting features for visual recognition problems. And in just the few years that followed, "deep learning" found great success in natural language processing, speech recognition, and reinforcement learning.

Any aspiring machine learning engineer should construct and train a deep convnet "from scratch."  Of course, there are varying degrees of "from scratch" - hence the quotes. I had already completed the exercise of implementing many of the neural network primitives using NumPy (e.g. fully connected layers, cross-entropy loss, batch normalization, LSTM / GRU cells, and convolution layers). So, here I use TensorFlow so I could focus on the challenges of training a large network on a large dataset with the benefit of efficiently coded GPU routines.

Amazingly, with 2 hours of GPU time (about $0.50 using an Amazon EC2 spot instance), it was not difficult to reach 50% top-1 accuracy and almost 80% top-5 accuracy. And, honestly, I was not able to identify many of the images that convnet got wrong (and even some of those it got correct).

## Dataset

Stanford prepared a [Tiny ImageNet](https://tiny-imagenet.herokuapp.com/) dataset for their [CS231n](http://cs231n.stanford.edu/) course. The dataset spans 200 image classes with 500 training examples per class. The dataset also has 50 validation and 50 test examples per class.

The images are down-sampled to 64x64 pixels vs. 256x256 for the full ImageNet dataset. The full ImageNet dataset also has 1000 classes. 

So, in short, I think Tiny ImageNet is large enough to be a challenging and realistic problem. Whereas a challenge like MNIST, CIFAR-10, or even CIFAR-100 are a bit too small (but obviously they are still very valuable datasets).

## Objectives

1. Successfully train a deep convnet
2. Implement saliency (i.e. where in the image is the model focused)
3. Perform kernel visualization
4. Experiment with alternative loss functions
    a. Smoothed cross-entropy loss
    b. SVM backend

For more details, see my blog: [Learning Artificial Intelligence](https://pat-coady.github.io).

## Quick Tour of Repository

### Python Files

**logistic_regression.py**

Always good practice to build a simple baseline to start. Gets about 3% top-1 classification accuracy (random guessing = 0.5%).

**single_layer_nn.py**

Another simple baseline. A neural net with a single hidden layer: 1024 hidden units with ReLU activations. Reaches about 8% accuracy without any attempt to tune it.

**vgg_16.py**

[This paper](https://arxiv.org/pdf/1409.1556.pdf) introduced the VGG-16 architecture. Karen Simonyan and Andrew Zisserman reached state-of-the-art performance using only a deep stack of 3x3xC filters and max-pooling layers. Because Tiny ImageNet has much lower resolution than the original ImageNet data, I removed the last max-pool layer and the last three convolution layers. With a little tuning, this model reaches 52% top-1 accuracy and 77% top-5 accuracy.

**input_pipe.py**

* Load JPEGs (using Tiny ImageNet directory structure)
* Load labels and build integer -> text dictionary
* QueueRunner to feed training
    * including data augmentation (i.e. various image distortions)

**train.py**

Trains models and monitors validation accuracy. The training loop has learning rate control and terminates training when progress stops. I take full advantage of TensorBoard by saving histograms of all weights, activations, and also learning curves.

Training is built to run fast on GPU by running the data pipeline on the CPU and model training on the GPU. It is straightforward to train a different model by changing 'model_name' in TrainConfig class.

**losses.py**

Contains three loss functions: 

1. Stanard cross-entropy loss
2. Smoothed cross-entropy loss (assign small, non-zero, probability to all classes)
3. SVM (works, but never got great performance)

**metrics.py**

Only one training metric right now: % accuracy.

### Notebooks

**predict_and_saliency.ipynb**

This is short notebook randomly selects ten images from the validation set and displays the top-5 predictions vs. the "gold" label. The notebook also displays saliency maps next to each image so you can see where the model is "looking" as it makes decisions.

**kernel_viz.ipynb**

Visualize input kernels (aka filters) of first two conv layers. The receptive field is only 7x7 after two 3x3 layers, but the results are still interesting.

**kernel_viz_conv4.ipynb**

Same as *kernel_viz.ipynb* except visualizes after 4th conv layer.

**val_accuracy.ipynb**

This notebook loads a model and calculates the validation set accuracy. It also computes the accuracy when predictions from 5 different crops x 2 flips are averaged: about a 3% accuracy improvement. This notebook runs very slow because it loops through the validation images one-by-one: It was not worth the extra effort to write efficiently. *Premature optimization is the root of all evil* --Donald Knuth

**image_distort.ipynb**

This short notebook displays images after TensorFlow applies image distortions. It is useful to see the distortions to bound how much hue and saturation to apply during training.

