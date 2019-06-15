---
layout: post
author: diogojc
title:  "Implementing a Recurring Neural Network for sentiment analysis"
image: "/assets/2018-09-08-artistic-style/images/thumbnail2.png"
excerpt: "Implementing a Recurring Neural Network for sentiment analysis"
description: "Implementing a Recurring Neural Network for sentiment analysis"
date:   2019-06-15
categories: [machine-learning, neural-networks, tensorflow]
tags: [tensorflow, neural networks, rnn, sentiment]
---

## Motivation for recurrent neural networks

With convolutional architectures we exploit spatial artifacts in data (e.g. lines, shapes, eyes) that can be learnt from other spatial artifacts that live in the vicinity. 

When you are dealing with sequential or temporal data this is typically not the case.

For example when learning structures on language certain artifacts (e.g. words) might only be relevant much later on. Although convolutional layers could technically be used they would struggle to learn these structures. Therefore a different type of architecture is used for sequential, or temporal, data.

In convolutional neural networks we learn weights in different layers representing a hierarchy of spatial concepts, in this new architecture the core idea is that we will learn a set of weights that for each timestep combine the data at that timestep with the memory of what has happened before. Notice that this set of weights is the same for every timestep. This looks very much like a Turing machine.

The image bellow shows this core idea. The "unrolled" representation on the right shows at each timestep $$t$$ the network will output some value $$h_t$$ that depends on the data $$X$$ at timestep $$t$$ ($$X_t$$) and the previous output $$h_{t-1}$$. $$A$$ is also called a cell and encapsulates both the learned weights and the operations that combine them with its inputs.

The left representation is a shorter representation that gives the network it's "recurrent" name. I find it more confusing then it's worth so I never use it.

{% include figure.html
           url="/assets/2019-06-15-sentiment-analysis/images/RNN-unrolled.png"
           description="https://colah.github.io/posts/2015-08-Understanding-LSTMs/"

%}

Forward and backward propagation move from left to right and right to left respectively "through time".

Cell $$A$$ should first and foremost easily propagate previous information. This is what allows these networks to learn quickly long term dependencies.

Cell $$A$$ will also contain whatever transformations necessary to figure out what the output should be at each timestep. It would even be possible to have stacked convolution operators inside such a cell to process sequences of images (i.e. video)!


## Objective

I will implement in Tensorflow a recurrent neural network that will learn to predict if a statement, in written text, is of positive or negative sentiment.

## Ground truth

To train this network I will use the [Stanford Sentiment Treebank][dataset-url] dataset. This contains statements about movies labeled as positive or negative.

{% include figure.html
           url="/assets/2019-06-15-sentiment-analysis/images/treebank.png"
           description="treebank example"

%}


## Methodology

At each timestep the cell can output some value. If you care to recognize entities in a piece of text you might want to output a mask for every word that identifies if a word is an entity or not. If you want to see if the sentiment of a sentence is positive or not you might only care for the output at the last timestep. If you want to translate a sentence to another language you might care in outputing multiple times after the last timestep.

These are the many-to-many, many-to-one and one-to-many configurations illustrated in the image bellow, that you want to use when specifying your loss function. 

{% include figure.html
           url="/assets/2019-06-15-sentiment-analysis/images/rnn-outputs.png"
           description="https://machinelearning-blog.com/2018/02/21/recurrent-neural-networks/"

%}


[colah-url]: https://colah.github.io/posts/2015-08-Understanding-LSTMs/
[dataset-url]: https://nlp.stanford.edu/sentiment/treebank.html