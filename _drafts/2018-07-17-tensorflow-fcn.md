---
layout: post
title:  "TensorFlow for Fully Convolutional Neural Networks"
description: ""
image: ""
date:   2018-07-17 15:49:00 +0200
categories: ai
tags: [tensorflow, neural networks, image, convolution, coco, segmentation, fcn]
---

## Motivation for Fully Convolutional Neural Networks

When I created a convolutional neural network for digit recognition at some point I flattened the 2D structure to a 1D structure representing the structure of the predictions that I cared about, which was a sequence from 0 to 9 (1D).

For other tasks the predictions I care about might have different dimensionalities. If I would like to know what class or classes each pixel belongs to I would be doing what is called image segmentation and the structure of predictions would resemble the same structure as the input image. This is the motivation for Fully Convolutional Neural Networks. At no point do we want to flatten the initial structure of the data and loose that information, therefore convolutions are used throught the network up to the prediction layer.

Bellow are illustrations from the [COCO Dataset][cocodataset-url] that show images overlayed with masks of different colors representing different instances of different classes. These are manually created and will serve as ground truth to create a neural network used for image segmentation.

![Cat segmentation]({{ "/assets/2018-07-17-tensorflow-fcn/images/segmentation1.png" | absolute_url }})

![People, dog and frisbee segmentation]({{ "/assets/2018-07-17-tensorflow-fcn/images/segmentation2.png" | absolute_url }})

To point out there is a difference between image segmentation, that is predicting which class every pixel belongs to, and instance segmentation, that is predicting which instance of which class every pixel belongs to. The image above makes the distinction between the two instances of dogs. In this post I will be doing image segmentation.


## U-Net architecture
[U-Net][u-net-paper-url] is a specific FCN architecture and the biggest differences from something like Lenet-5 is that it contains an encoder and a decoder parts.
By encoding I mean the architecture first (through a series of convolutions) finds a representation of the input in a set of encodings of smaller dimensionalities. Afterwards the decoding part (through a series of transposed convolutions) brings these encodings to the original dimensionality.

The way I like to think about this type of architecture is the network is forcing the original image to be reduced to its essential components first and afterwards building on top of these learned encodings to create predictions.

Bellow is an illustration from the paper where you can see the left and right part as the encoding and decoding bits forming a U shape.

![u-net]({{ "/assets/2018-07-17-tensorflow-fcn/images/u-net-architecture.png" | absolute_url }})

To learn more about this architecture go to through the [website][u-net-website-url] or [article][u-net-paper-url].

A remark on the "up convolution" (or transposed convolution) operator in the network. This is actually a normal convolution that is built in a way that the output is actually bigger then the input.
Bellow is a "normal convolution" illustration taken from [here][convolutions-url]. The cyan


<table style="width:100%; table-layout:fixed;">
  <tr>
    <td>
        ![Convolution]({{ "https://raw.githubusercontent.com/vdumoulin/conv_arithmetic/master/gif/no_padding_no_strides.gif" }})
    </td>
    <td>
        ![Transposed Convolution]({{ "https://raw.githubusercontent.com/vdumoulin/conv_arithmetic/master/gif/no_padding_no_strides_transposed.gif" }})
    </td>
  </tr>
  <tr>
    <td>No padding, no strides</td>
    <td>Arbitrary padding, no strides</td>
  </tr>
</table>


Although this is not how it is computed a way to understand what a transposed convolution is to imagine a normal convolution with padding. This makes the output bigger then the actual input 








## COCO dataset

## Feeding COCO data to Tensorflow

When I had smaler datasets I would create a Placeholder Tensor and fill that when running the graph with whatever I had in the memory of my python script usign the feed_dict parameter.
This going back and forth forces the python runtime and tensorflow to exchange content creating copies in memory and slowing down computation. For such a neural network on images this becomes quite a bottleneck so this mechanism of feeding the values of the placeholders to tensorflow at run time cannot be used.

This is the reason of being of the TFRecords. These are files serialized using Protobuf that will be hooked up directly to the computation graph using multiple threads and a queueing mechanism connecting them to the disk.

### Creating TFRecords
{% highlight python %}
writer = tf.python_io.TFRecordWriter(outFile)
i = 1
for imgId in imgIds:
    print("{:10.2f}%".format(100*(float)(i)/len(imgIds)), end='\r')
    image = next(self.getImagesFromImgIds(imgId))[..., ::-1]
    mask = self.getBinaryMaskFromImgId(imgId, classNames)
    features = {
        "imageContent": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tostring()])),
        "maskContent": tf.train.Feature(bytes_list=tf.train.BytesList(value=[mask.tostring()])),
        "height": tf.train.Feature(int64_list=tf.train.Int64List(value=[image.shape[0]])),
        "width": tf.train.Feature(int64_list=tf.train.Int64List(value=[image.shape[1]])),
    }
    example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(example.SerializeToString())
    i += 1
writer.close()
{% endhighlight %}

### Connecting TFRecords to the Tensorflow graph.
{% highlight python %}
def GetDataIterator(tfrecordPath):
    def decode(serialized_example):
        features = {
            "imageContent": tf.FixedLenFeature([], tf.string),
            "maskContent": tf.FixedLenFeature([], tf.string),
            "height": tf.FixedLenFeature([], tf.int64),
            "width": tf.FixedLenFeature([], tf.int64),
        }
        example = tf.parse_single_example(serialized_example, features)

        imageContent = tf.decode_raw(example['imageContent'], tf.uint8)
        maskContent = tf.decode_raw(example['maskContent'], tf.uint8)
        height = example['height']
        width = example['width']
        imageShape = tf.stack([height, width, 3])
        maskShape = tf.stack([height, width, 1])
        image = tf.reshape(imageContent, imageShape)
        mask = tf.reshape(maskContent, maskShape)
        resizedImage = tf.image.resize_image_with_crop_or_pad(image, 572, 572)
        resizedMask = tf.image.resize_image_with_crop_or_pad(mask, 388, 388)
        return (tf.cast(resizedImage, tf.float32),
                tf.cast(resizedMask, tf.float32))

    dataset = tf.data.TFRecordDataset(tfrecordPath)
    dataset = dataset.map(decode)
    batch_size = 7
    batch = dataset.batch(batch_size)
    iterator = batch.make_initializable_iterator()
    return iterator
{% endhighlight %}

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

You can checkout the entire source code [here][fcncode-url].


[cocodataset-url]: http://cocodataset.org/
[u-net-paper-url]: https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
[u-net-website-url]: https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
[convolutions-url]: https://github.com/vdumoulin/conv_arithmetic
[fcncode-url]: https://github.com/diogojc/diogojc.github.io/blob/master/assets/2018-07-17-tensorflow-fcn/code/fcn.py