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

Since we want to classify every pixel in an image our output will be similar to the input image but instead of colour (grayscale) channel(s) there will be probabilities associated with the possible classes.

A typical approach here are fully convolutional networks (FCNs). The name comes from the fact that (loosely speaking) activations are only done using the convolution operation. This is so we maintain the original structure of the input up to and including the output layer.

## U-Net architecture
[U-Net][u-net-url] is a specific FCN architecture and the biggest differences from something like Lenet-5 is the up-sampling part of the architecture and the reusing of activations in the down-sampling part.
This down and up-sampling creates a (albeit contrived) U shaped architecture like bellow.

![u-net]({{ "/assets/2018-07-17-tensorflow-fcn/images/u-net-architecture.png" | absolute_url }})

TODO: explain up sampling (use this https://github.com/vdumoulin/conv_arithmetic)

## COCO dataset

### Converting to TFRecord
dasdasas

### Feeding TFRecord to Tensorflow

When I had small datasets I would create a Placeholder Tensor and fill that when running the graph with whatever I had in the memory of my python script usign the feed_dict parameter.
This going back and forth 

## Tensorflow model

{% highlight python %}
def convolutions(inputLayer, numChannels):
    conv_1 = tf.layers.conv2d(inputLayer,
                              numChannels,
                              [3, 3],
                              padding="valid",
                              activation=tf.nn.relu)
    batchnorm_1 = tf.layers.batch_normalization(conv_1, training=True)
    conv_2 = tf.layers.conv2d(batchnorm_1,
                              numChannels,
                              [3, 3],
                              padding="valid",
                              activation=tf.nn.relu)
    batchnorm_2 = tf.layers.batch_normalization(conv_2, training=True)
    return batchnorm_2
{% endhighlight %}

{% highlight python %}
def maxPool(inputLayer):
    return tf.layers.max_pooling2d(inputLayer, 2, 2, padding='valid')
{% endhighlight %}

{% highlight python %}
def upConvolution(inputLayer, numChannels):
    upconv = tf.layers.conv2d_transpose(inputLayer,
                                        numChannels,
                                        [2, 2],
                                        strides=[2, 2],
                                        padding='valid')
    batchnorm = tf.layers.batch_normalization(upconv, training=True)
    return batchnorm
{% endhighlight %}

{% highlight python %}
def MergeLayers(inputLayer, siblingLayer):
    inputLayerShape = inputLayer.shape.as_list()[1]
    siblingLayerShape = siblingLayer.shape.as_list()[1]
    begincrop = (int)((siblingLayerShape - inputLayerShape)/2)
    croppedSiblingLayer = slice(siblingLayer,
                                [0, begincrop, begincrop, 0],
                                [-1, inputLayerShape, inputLayerShape, -1])
    mergedLayer = concat([croppedSiblingLayer, inputLayer], 3)
    return mergedLayer
{% endhighlight %}

{% highlight python %}
def UNet(X):
    conv1 = convolutions(X, 64)
    maxpool1 = maxPool(conv1)
    conv2 = convolutions(maxpool1, 128)
    maxpool2 = maxPool(conv2)
    conv3 = convolutions(maxpool2, 256)
    maxpool3 = maxPool(conv3)
    conv4 = convolutions(maxpool3, 512)
    maxpool4 = maxPool(conv4)
    conv5 = convolutions(maxpool4, 1024)

    upconv1 = upConvolution(conv5, 512)
    merge1 = MergeLayers(upconv1, conv4)
    conv6 = convolutions(merge1, 512)
    upconv2 = upConvolution(conv6, 256)
    merge2 = MergeLayers(upconv2, conv3)
    conv7 = convolutions(merge2, 256)
    upconv3 = upConvolution(conv7, 128)
    merge3 = MergeLayers(upconv3, conv2)
    conv8 = convolutions(merge3, 128)
    upconv4 = upConvolution(conv8, 64)
    merge4 = MergeLayers(upconv4, conv1)
    conv9 = convolutions(merge4, 64)

    return tf.layers.conv2d(conv9,
                            1,
                            [1, 1],
                            padding="valid",
                            activation=tf.sigmoid)
{% endhighlight %}

### Loss function

{% highlight python %}
cost = tf.reduce_sum(tf.keras.backend.binary_crossentropy(y, lu1_4))
{% endhighlight %}

### Tensorboard graph
![tensorflow graph]({{ "/assets/2018-07-17-tensorflow-fcn/images/unet_tensorboard.png" | absolute_url }})

## Training
![input]({{ "/assets/2018-07-17-tensorflow-fcn/images/cat.png" | absolute_url }})
![prediction]({{ "/assets/2018-07-17-tensorflow-fcn/images/catPrediction.png" | absolute_url }})



[u-net-url]: https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/