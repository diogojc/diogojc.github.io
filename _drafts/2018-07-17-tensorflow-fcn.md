---
layout: post
title:  "TensorFlow for Fully Convolutional Neural Networks"
description: ""
image: ""
date:   2018-07-17 15:49:00 +0200
categories: ai
tags: [tensorflow, neural networks, image, convolution, coco, segmentation, fcn]
---

## Motivation for image segmentation

When I created a convolutional neural network for digit recognition at some point I flattened the 2D structure to correspond to a 1D structure representing possible classifications for the whole image. In some cases this is not enough because we might want to know which pixels belong to which classes. This type of problem is called image segmentation and a new type of architecture is needed.

Since we want classify every pixel in an image our output will be similar to the input image but instead of colour (grayscale) channel(s) there will be probabilities associated with the possible classes.

A typical approach here are fully convolutional networks (FCNs). The name comes from the fact that (loosely speaking) activations are only done using the convolution operation. This is so we maintain the original structure of the input up to and including the output layer.

## U-Net architecture
[U-Net][u-net-url] is a specific FCN architecture and the biggest differences from something like Lenet-5 is the up-sampling part of the architecture and the reusing of activations in the down-sampling part.
This down and up-sampling creates a (albeit contrived) U shaped architecture like bellow.

![u-net]({{ "/assets/2018-07-17-tensorflow-fcn/images/u-net-architecture.png" | absolute_url }})

TODO: explain up sampling (use this https://github.com/vdumoulin/conv_arithmetic)

## Example image segmentation with COCO dataset


### Loading COCO dataset
dasdasas

[u-net-url]: https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/