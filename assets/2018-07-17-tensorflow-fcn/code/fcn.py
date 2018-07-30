import tensorflow as tf
from tensorflow import concat, slice
from tensorflow.python import debug as tf_debug


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
    return tf.cast(resizedImage, tf.float32), tf.cast(resizedMask, tf.float32)

dataset = tf.data.TFRecordDataset("/Users/diogoc/Downloads/coco/val2017.tfrecord")
dataset = dataset.map(decode)
epochs = 10
batch_size = 5
batch = dataset.batch(batch_size)
iterator = batch.make_initializable_iterator()
next_element = iterator.get_next()
X, y = next_element




# down path
ld1_1 = tf.layers.batch_normalization(X, training=True)
ld1_2 = tf.layers.conv2d(ld1_1, 64, [3, 3], padding="valid", activation=tf.nn.relu)
ld1_3 = tf.layers.conv2d(ld1_2, 64, [3, 3], padding="valid", activation=tf.nn.relu)

ld2_1 = tf.layers.max_pooling2d(ld1_3, 2, 2, padding='valid')
ld2_2 = tf.layers.conv2d(ld2_1, 128, [3, 3], padding="valid", activation=tf.nn.relu)
ld2_3 = tf.layers.conv2d(ld2_2, 128, [3, 3], padding="valid", activation=tf.nn.relu)

ld3_1 = tf.layers.max_pooling2d(ld2_3, 2, 2, padding='valid')
ld3_2 = tf.layers.conv2d(ld3_1, 256, [3, 3], padding="valid", activation=tf.nn.relu)
ld3_3 = tf.layers.conv2d(ld3_2, 256, [3, 3], padding="valid", activation=tf.nn.relu)

ld4_1 = tf.layers.max_pooling2d(ld3_3, 2, 2, padding='valid')
ld4_2 = tf.layers.conv2d(ld4_1, 512, [3, 3], padding="valid", activation=tf.nn.relu)
ld4_3 = tf.layers.conv2d(ld4_2, 512, [3, 3], padding="valid", activation=tf.nn.relu)

l5_1 = tf.layers.max_pooling2d(ld4_3, 2, 2, padding='valid')
l5_2 = tf.layers.conv2d(l5_1, 1024, [3, 3], padding="valid", activation=tf.nn.relu)
l5_3 = tf.layers.conv2d(l5_2, 1024, [3, 3], padding="valid", activation=tf.nn.relu)


# up path
upsamp = tf.layers.conv2d_transpose(l5_3, 512, [2, 2], strides=[2, 2], padding='valid')
copy = ld4_3
begincrop = (int)((copy.shape.as_list()[1] - upsamp.shape.as_list()[1])/2)
sizecrop = upsamp.shape.as_list()[1]
copycrop = slice(copy, [0, begincrop, begincrop, 0], [-1, sizecrop, sizecrop, -1])
lu4_1 = concat([copycrop, upsamp], 3)
lu4_2 = tf.layers.conv2d(lu4_1, 512, [3, 3], padding="valid", activation=tf.nn.relu)
lu4_3 = tf.layers.conv2d(lu4_2, 512, [3, 3], padding="valid", activation=tf.nn.relu)

upsamp = tf.layers.conv2d_transpose(lu4_3, 256, [2, 2], strides=[2, 2], padding='valid')
copy = ld3_3
begincrop = (int)((copy.shape.as_list()[1] - upsamp.shape.as_list()[1])/2)
sizecrop = upsamp.shape.as_list()[1]
copycrop = slice(copy, [0, begincrop, begincrop, 0], [-1, sizecrop, sizecrop, -1])
lu3_1 = concat([copycrop, upsamp], 3)
lu3_2 = tf.layers.conv2d(lu3_1, 256, [3, 3], padding="valid", activation=tf.nn.relu)
lu3_3 = tf.layers.conv2d(lu3_2, 256, [3, 3], padding="valid", activation=tf.nn.relu)

upsamp = tf.layers.conv2d_transpose(lu3_3, 128, [2, 2], strides=[2, 2], padding='valid')
copy = ld2_3
begincrop = (int)((copy.shape.as_list()[1] - upsamp.shape.as_list()[1])/2)
sizecrop = upsamp.shape.as_list()[1]
copycrop = slice(copy, [0, begincrop, begincrop, 0], [-1, sizecrop, sizecrop, -1])
lu2_1 = concat([copycrop, upsamp], 3)
lu2_2 = tf.layers.conv2d(lu2_1, 128, [3, 3], padding="valid", activation=tf.nn.relu)
lu2_3 = tf.layers.conv2d(lu2_2, 128, [3, 3], padding="valid", activation=tf.nn.relu)

upsamp = tf.layers.conv2d_transpose(lu2_3, 64, [2, 2], strides=[2, 2], padding='valid')
copy = ld1_3
begincrop = (int)((copy.shape.as_list()[1] - upsamp.shape.as_list()[1])/2)
sizecrop = upsamp.shape.as_list()[1]
copycrop = slice(copy, [0, begincrop, begincrop, 0], [-1, sizecrop, sizecrop, -1])
lu1_1 = concat([copycrop, upsamp], 3)
lu1_2 = tf.layers.conv2d(lu1_1, 64, [3, 3], padding="valid", activation=tf.nn.relu)
lu1_3 = tf.layers.conv2d(lu1_2, 64, [3, 3], padding="valid", activation=tf.nn.relu)
lu1_4 = tf.layers.conv2d(lu1_3, 1, [1, 1], padding="valid", activation=tf.nn.relu)

# caca = tf.reduce_all(tf.logical_not(tf.is_nan(lu1_4)))
# assert_op = tf.Assert(caca, [lu1_4])

# loss function
flat_lu1_4 = tf.reshape(lu1_4, [-1, 388*388])
flat_y = tf.reshape(y, [-1, 388*388])
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_lu1_4, labels=flat_y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

tf.summary.scalar('cross_entropy', cost)
tf.summary.image("trainingImages", X, max_outputs=batch_size)
tf.summary.image("trainingMasks", y, max_outputs=batch_size)
summaryOp = tf.summary.merge_all()

# https://wookayin.github.io/tensorflow-talk-debugging/#40
with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    # sess = tf_debug.TensorBoardDebugWrapperSession(sess, "localhost:6064")
    writer = tf.summary.FileWriter('./logs/train ', sess.graph)
    sess.graph.finalize()
    for epoch in range(epochs):
        print("Starting epoch {}...".format(epoch))
        sess.run(iterator.initializer)
        try:
            batchnr = 1
            while True:
                summary, c = sess.run([summaryOp, cost])
                writer.add_summary(summary, batchnr)
                #if batchnr % batch_size:
                print("Epoch: {}, Batch: {}, Cost: {}".format(epoch, batchnr, c))
                batchnr += 1
        except tf.errors.OutOfRangeError:
            pass
