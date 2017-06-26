# Tiny ImageNet

## Introduction

I doubt that is controversial to say that [ImageNet](http://www.image-net.org/) and Alex Krizhevsky's ["AlexNet"](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) sparked a revolution in machine learning. By and large, AlexNet marked the end of the era of hand-crafting features for visual recognition problems. And in just the few years that followed, "deep learning" found great success in natural language processing, speech recognition and reinforcement learning.

Any aspiring machine learning engineer should contruct and train a very deep convnet "from scratch."  Of course, there are varying degrees of "from scratch" - hence the quotes. I had already completed the exercise of implementing many of the neural network primitives using NumPy (e.g. fully conntected layers, cross-entropy loss, batch normalization, LSTM / GRU cells, and convolution layers). So, here I use TensorFlow so I could focus on the challenges of training a large network on a large dataset with the benefit of efficiently coded GPU routines.

Amazingly, with 2 hours of GPU time (about $0.50 using an Amazon EC2 spot instance), it was not difficult to reach 50% top-1 accuracy and almost 80% top-5 accuracy. And, honestly, I was not able to identify many of the images that convnet got wrong (and even some of those it got correct).

## Dataset

Stanford prepared a [Tiny ImageNet](https://tiny-imagenet.herokuapp.com/) dataset for their [CS231n](http://cs231n.stanford.edu/) course. The dataset spans 200 image classes with 500 training examples per class. The dataset also has 50 validation and 50 test examples per class.

The images are down-sampled to 64x64 pixels vs. 256x256 for the full ImageNet dataset. The full ImageNet dataset also has 1000 classes. 

So, in short, I think Tiny ImageNet is large enough to be a challenging and realistic problem. Whereas a challenge like MNIST, CIFAR-10, or even CIFAR-100 are a bit too small (but obviously they are still very important datasets).

## Objectives

1. Succesfully train a deep convnet
2. Implement saliency (i.e. where in the image is the model focused)
3. Implement kernel visualization
4. Experiment with alternative loss functions
	a. Smoothed cross-entropy loss
	b. SVM backend

For more details, see my blog: [Learning Artificial Intelligence](https://pat-coady.github.io).

## Quick Tour of Repository

### .py files

**input_pipe.py:**

* Load JPEGs (using Tiny ImageNet directory structure)
* Load labels and build integer -> text dictionary
* QueueRunner to feed training
	* including data augmentation (i.e. various image distortions)

**logistic_regression.py:**

Always good practice to build a simple baseline to start. Gets about 3% top-1 classification accuracy (random guessing = 0.5%).

**single_layer_nn.py:**

Another simple baseline. A neural net with a single hidden layer: 1024 hidden units with ReLU activations. Reaches about 8% accuracy without any attempt to tune it.

**vgg_16.py**

This is the deep convnet I focused on. The architecture is based on VGG-16 from [this paper](https://arxiv.org/pdf/1409.1556.pdf). The authors (Karen Simonyan and Andrew Zisserman reached state-of-the-art performance using only a deep stack of 3x3 filters and max-pooling layers. Because Tiny ImageNet has much lower resolution, I removed the last max-pool layer and the last 3 convolution layers. With very little tuning this model reaches 52% top-1 accuracy and 77% top-5 accuracy.



### notebooks



