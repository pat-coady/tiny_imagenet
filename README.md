# Tiny ImageNet

## Introduction

[ImageNet](http://www.image-net.org/) and Alex Krizhevsky's ["AlexNet"](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) sparked a revolution in machine learning. AlexNet marked the end of the era of mostly hand-crafted features for visual recognition problems. In just the few years that followed AlexNet, "deep learning" found great success in natural language processing, speech recognition, and reinforcement learning.

Any aspiring machine learning engineer should construct and train a deep convnet "from scratch."  Of course, there are varying degrees of "from scratch." I had already implemented many of the neural network primitives using NumPy (e.g. fully connected layers, cross-entropy loss, batch normalization, LSTM / GRU cells, and convolutional layers). So, here I use TensorFlow so the focus is on training a deep network on a large dataset.

Amazingly, with only 2 hours of GPU time (about $0.50 using an Amazon EC2 spot instance), it was not difficult to reach 50% top-1 accuracy and almost 80% top-5 accuracy. At this accuracy, I was also making mistakes on the images that the model got wrong (and I even made mistakes on some that it got correct).

## Dataset

Stanford prepared the [Tiny ImageNet](https://tiny-imagenet.herokuapp.com/) dataset for their [CS231n](http://cs231n.stanford.edu/) course. The dataset spans 200 image classes with 500 training examples per class. The dataset also has 50 validation and 50 test examples per class.

The images are down-sampled to 64x64 pixels vs. 256x256 for the original ImageNet. The full ImageNet dataset also has 1000 classes. 

Tiny ImageNet is large enough to be a challenging and realistic problem. But not so big as to require days of training before you see results.

## Objectives

1. Train a high-performance deep CNN
2. Implement saliency (i.e. Where in the image is the model focused?)
3. Visualize convolution filters
4. Experiment with alternative loss functions
    a. Smoothed cross-entropy loss
    b. SVM

For more details, see my blog: [Learning Artificial Intelligence](http://learningai.io).

## Quick Tour of Repository

### Python Files

**logistic_regression.py**

It is good practice to build a simple baseline to start. This baseline gets reaches around 3% top-1 classification accuracy (random guessing = 0.5%).

**single_layer_nn.py**

Another simple baseline. A neural net with a single hidden layer: 1024 hidden units with ReLU activations. Reaches about 8% accuracy with minimal tuning effort.

**vgg_16.py**

[This paper](https://arxiv.org/pdf/1409.1556.pdf) by Karen Simonyan and Andrew Zisserman introduced the VGG-16 architecture. The authors reached state-of-the-art performance using only a deep stack of 3x3xC filters and max-pooling layers. Because Tiny ImageNet has much lower resolution than the original ImageNet data, I removed the last max-pool layer and the last three convolution layers. With a little tuning, this model reaches 52% top-1 accuracy and 77% top-5 accuracy.

To keep it fair, I didn't use any pre-trained VGG-16 layers and only trained using the Tiny ImageNet examples.

**input_pipe.py**

* Load JPEGs (using Tiny ImageNet directory structure)
* Load labels and build label integer-to-text dictionary
* QueueRunner to feed GPU
    * including data augmentation (i.e. various image distortions)

**train.py**

Trains models and monitors validation accuracy. The training loop has learning rate control and terminates training when progress stops. I take full advantage of TensorBoard by saving histograms of all weights, activations, and also learning curves.

Training is built to run fast on GPU by running the data pipeline on the CPU and model training on the GPU. It is straightforward to train a different model by changing 'model_name' in TrainConfig class.

**losses.py**

Contains three loss functions: 

1. Cross-entropy loss
2. Smoothed cross-entropy loss (add small, non-zero, probability to all classes)
3. SVM (works, but never got great performance)

**metrics.py**

Measures % accuracy.

### Notebooks

**predict_and_saliency.ipynb**

This short notebook randomly selects ten images from the validation set and displays the top-5 predictions vs. the "gold" label. The notebook also displays saliency maps next to each image so you can see where the model is "looking" as it makes decisions.

**kernel_viz.ipynb**

Visualize input kernels (aka filters) of first two conv layers. The receptive field is only 7x7 after two 3x3 layers, but the results are still interesting.

**kernel_viz_conv4.ipynb**

Same as *kernel_viz.ipynb* except visualizes after 4th conv layer.

**val_accuracy.ipynb**

This notebook loads a model and calculates the validation set accuracy. It also computes the accuracy when predictions from 5 different crops x 2 flips are averaged: about a 3% accuracy improvement. This notebook runs slowly because it loops through the validation images one-by-one: It was not worth the extra effort to write efficiently. *Premature optimization is the root of all evil.* -Donald Knuth

**image_distort.ipynb**

This short notebook displays images after TensorFlow applies image distortions. It is useful to see the distortions to bound the random hue and saturation distortions applied during training.
