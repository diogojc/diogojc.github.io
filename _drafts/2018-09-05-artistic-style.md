---
layout: post
author: diogojc
title:  "Implementing a neural algorithm of artistic style"
image: "/assets/2018-09-05-artistic-style/images/stylizedContent.png"
excerpt: "Implementing a neural algorithm for artistic style."
description: "Implementing a neural algorithm for artistic style."
date: 2018-09-05
categories: [machine-learning, neural-networks, tensorflow]
tags: [tensorflow, neural networks, image, convolution, artistic, style, creation, creative]
---


## Motivation

The motivation for this post is threefold:
* Using these networks to, instead of describing what exists, creating novel things.
* Show how unresanably good CNNs are at learning fine and coarsed grained representations of spatial data.
* Demonstrate how flexible Tensorflow is as (described [here]({% post_url 2018-05-01-tensorflow-basics %})) an optimization mechanism for arbitrary computation graphs.

{% include figures.html
           url1="/assets/2018-09-05-artistic-style/images/content.jpg"
           url2="/assets/2018-09-05-artistic-style/images/style.jpg"
           description="Image to apply style (left) and image containing style (right)"
%}

{% include figure.html
           url="/assets/2018-09-05-artistic-style/images/stylizedContent.png"
           description="Stylized content"
%}

The code excerpts bellow were taken from the [complete source code][source-code-url].

## Objective

For simplicity I focused on the [original paper][original-paper-url] that showed these types of approach work. But you can find much more faster/complete implementations of further aditions to the original work.


## Methodology

We start with some pre-trained layers of the VGG19 model and attach some new loss functions

### Content
Explain in plain terms what this is. 
Show briefly that a hierarchy of concepts is learned in the VGG19 and we use it to compare if "high level" representations of the objects in the content image are also present in the stylized image.

$$L_{content}(\vec{p},\vec{x},l) =\frac{1}{2}\sum_{i,j}(F_{ij}^l - P_{ij}^l)^2$$

I used the [L2 loss in Tensorflow][l2-loss-url]

{% highlight python %}
def ContentLoss(F, P):
    return tf.nn.l2_loss(F - P)
{% endhighlight %}

### Style 
Explain in plain terms what this is.
Show this build on top of the "high level" representations of the objects present in the style image and that gram matrices just have values of how much certain features co-occur a lot. This so the stylized image also contains these co-ocurrences.

The formulas bellow were taken from the [original paper][original-paper-url].

$$G_{ij}^l = \sum_{k}F_{ik}^l F_{jk}^l$$

{% highlight python %}
def GramMatrix(layer):
    _, h, w, c = layer.get_shape().as_list()
    F = tf.reshape(layer, [h*w, c])
    return tf.matmul(tf.transpose(F), F)
{% endhighlight %}

$$E_{l} =\frac{1}{4N_{l}^2M_{l}^2}\sum_{i,j}(G_{ij}^l - A_{ij}^l)^2$$

{% highlight python %}
def StyleLoss(A, G, M, N):
    return tf.nn.l2_loss(G - A) / (2*N**2*M**2)
{% endhighlight %}

$$L_{style}(\vec{a},\vec{x}) = \sum_{l=0}^Lw_{l}E_{l}$$

{% highlight python %}
...
styleLoss = reduce(tf.add,
                   [StyleLoss(A[l],
                              G[l],
                              M[l],
                              N[l]) for l in STYLE_LAYERS])
styleLoss /= len(STYLE_LAYERS)
...
{% endhighlight %}

### Loss function

$$L(\vec{p},\vec{a},\vec{x}) = \alpha L_{content}(\vec{p},\vec{x}) + \beta L_{style}(\vec{a},\vec{x})$$

{% highlight python %}
alpha = 10e-4
beta = 1
loss = alpha * contentLoss + beta * styleLoss
{% endhighlight %}

### Pre-trained network

{% highlight python %}
def getWeights():
    with tf.Graph().as_default():
        weights = VGG19(weights='imagenet', include_top=False).get_weights()
    return weights
{% endhighlight %}

{% highlight python %}
def getActivations(X, weights, debug=False):
    layers = [2, 2, 4, 4, 1]
    activations = {}
    w = 0
    for i in range(len(layers)):
        for j in range(layers[i]):
            with tf.name_scope("block{}_{}".format(i+1, j+1)):
                conv_W = tf.constant(weights[w], name="W")
                conv_b = tf.constant(weights[w+1], name="b")
                conv = tf.nn.conv2d(X,
                                    conv_W,
                                    strides=[1, 1, 1, 1],
                                    padding='SAME') + conv_b
                relu = tf.nn.relu(conv)
                activations["relu{}_{}".format(i+1, j+1)] = relu
                if debug:
                    tf.summary.histogram("weights", conv_W)
                    tf.summary.histogram("bias", conv_b)
                    tf.summary.histogram("relu", relu)
                X = relu
                w += 2
        if i+1 is not 5:
            with tf.name_scope("pool{}".format(i+1)):
                X = tf.nn.pool(X, [2, 2], "AVG", "VALID", strides=[2, 2])
    return activations
{% endhighlight %}

### Training




[original-paper-url]: https://arxiv.org/abs/1508.06576
[source-code-url]: https://github.com/diogojc/diogojc.github.io/tree/master/assets/2018-09-05-artistic-style/code
[l2-loss-url]: https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss
