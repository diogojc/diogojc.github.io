import tensorflow as tf
from tensorflow import concat, slice


def GetDataIterator(tfrecordPath, batchSize=7):
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
    batch = dataset.batch(batchSize)
    iterator = batch.make_initializable_iterator()
    return iterator


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


def maxPool(inputLayer):
    return tf.layers.max_pooling2d(inputLayer, 2, 2, padding='valid')


def upConvolution(inputLayer, numChannels):
    upconv = tf.layers.conv2d_transpose(inputLayer,
                                        numChannels,
                                        [2, 2],
                                        strides=[2, 2],
                                        padding='valid')
    batchnorm = tf.layers.batch_normalization(upconv, training=True)
    return batchnorm


def mergeLayers(inputLayer, siblingLayer):
    inputLayerShape = inputLayer.shape.as_list()[1]
    siblingLayerShape = siblingLayer.shape.as_list()[1]
    begincrop = (int)((siblingLayerShape - inputLayerShape)/2)
    croppedSiblingLayer = slice(siblingLayer,
                                [0, begincrop, begincrop, 0],
                                [-1, inputLayerShape, inputLayerShape, -1])
    mergedLayer = concat([croppedSiblingLayer, inputLayer], 3)
    return mergedLayer


def outputLayer(inputLayer, numChannels):
    return tf.layers.conv2d(inputLayer,
                            numChannels,
                            [1, 1],
                            padding="valid",
                            activation=tf.sigmoid)


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
    merge1 = mergeLayers(upconv1, conv4)
    conv6 = convolutions(merge1, 512)
    upconv2 = upConvolution(conv6, 256)
    merge2 = mergeLayers(upconv2, conv3)
    conv7 = convolutions(merge2, 256)
    upconv3 = upConvolution(conv7, 128)
    merge3 = mergeLayers(upconv3, conv2)
    conv8 = convolutions(merge3, 128)
    upconv4 = upConvolution(conv8, 64)
    merge4 = mergeLayers(upconv4, conv1)
    conv9 = convolutions(merge4, 64)

    return outputLayer(conv9, 1)


tf.reset_default_graph()
iterator = GetDataIterator("/Users/diogoc/Downloads/coco/val2017.tfrecord",
                           batchSize=7)
X, y = iterator.get_next()
y_hat = UNet(X)
cost = tf.reduce_sum(tf.keras.backend.binary_crossentropy(y, y_hat))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
tf.summary.scalar("cost", cost)
tf.summary.image("trainingImages", X, max_outputs=2)
tf.summary.image("trainingMasks", y, max_outputs=2)
tf.summary.image("trainingPred", y_hat, max_outputs=2)
summaryOp = tf.summary.merge_all()
config = tf.ConfigProto(device_count={'GPU': 1}, log_device_placement=True)
with tf.Session(config=config) as sess:
    sess.run([
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
        ])
    writer = tf.summary.FileWriter("logs", sess.graph)
    sess.graph.finalize()
    epochs = 500
    runnr = 1
    for epoch in range(1, epochs+1):
        print("Starting epoch {}...".format(epoch))
        sess.run(iterator.initializer)
        try:
            batchnr = 1
            while True:
                _, c, summary = sess.run([optimizer, cost, summaryOp])
                writer.add_summary(summary, runnr)
                print("Run: {}, Epoch: {}, Batch: {}, Cost: {}".format(runnr,
                                                                       epoch,
                                                                       batchnr,
                                                                       c))
                batchnr += 1
                runnr += 1
        except tf.errors.OutOfRangeError:
            pass
