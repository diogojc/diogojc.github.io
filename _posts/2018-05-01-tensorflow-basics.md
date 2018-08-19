---
layout: post
author: diogojc
title:  "TensorFlow basics"
image: "/assets/2018-05-01-tensorflow-basics/images/banner.jpg"
excerpt: "Tensorflow introduction as an optimization library for arbitrary computation graphs."
description: "Tensorflow introduction as an optimization library for arbitrary computation graphs."
date:   2018-05-01 21:38:52 +0200
categories: ai
tags: [tensorflow, neural networks, optimization]
---
In this post I want to lay out the fundamentals for my understanding of how to work further with [TensorFlow][tensorflow-url].

As far as I understand TensorFlows biggest proposition is that you can find the parameters that miminize/maximize a given function without having to specify it's gradients.
When using TensorFlow to train neural networks this is particularly handy for the backpropagation step with custom activation and cost functions.

# Importing TensorFlow

The first thing necessary is to load the library. I prefer to use the shorter 'tf' alias.
{% highlight python %}
import tensorflow as tf
{% endhighlight %}

# Evaluating a function

Before trying to optimize functions lets just try to represent a function with parameters and evaluate it.
Take the $$y$$ function bellow:

$$y(x) = 2*x + 4$$

To represent this in TensorFlow we would create two constants, a parameter and a function that ties it all together like so: 

{% highlight python %}
x = tf.get_variable("x", [1])
y = 2*x + 4
{% endhighlight %}

In TensorFlow lingo we have just defined a computation graph. To evaluate $$y$$ at $$x=5$$ we would:

{% highlight python %}
sess = tf.Session()
sess.run(x.assign([5]))
y_at_5 = sess.run(y)
print(y_at_5)
{% endhighlight %}

This should produce:

{% highlight bash %}
[14.]
{% endhighlight %}

# Finding the parameters that minimize a function

Let's now consider the following function:

$$y=\left(\frac{x}{4}-3\right)^2$$

In TensorFlow we create the corresponding computation grah like so:

{% highlight python %}
x = tf.get_variable("x", [1])
y = tf.square(x/4-3)
{% endhighlight %}

What if we are interested to find the value of $$x$$ that minimizes $$y$$ using TensorFlow?
Tensorflow comes with optimizers that will search the function for the minimum using the gradients automatically created from the computation graph.
We just have to initialize $$x$$ at an arbitrary location and iterate through the optimization operation that updates $$x$$ closer to the minimum.

{% highlight python %}
optimize = tf.train.GradientDescentOptimizer(0.01).minimize(y)
sess = tf.Session();
init = tf.global_variables_initializer()
sess.run(init)
for i in range(5000):
    sess.run(optimize)
print(sess.run(x))
{% endhighlight %}

We see the number bellow which is the value for $$x$$ that TensorFlow found to be approaching the minimum for $$y$$.
{% highlight bash %}
[11.97526]
{% endhighlight %}

# Next
[Next]({{ site.baseurl }}{% post_url 2018-05-06-tensorflow-nn %}) I will write how to use this basis to train a neural network.

[tensorflow-url]: https://www.tensorflow.org/

