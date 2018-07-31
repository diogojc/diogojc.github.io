import tensorflow as tf
from tensorflow import concat, slice
from tensorflow.python import debug as tf_debug

tf.reset_default_graph()

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

dataset = tf.data.TFRecordDataset("train2017.tfrecord")
dataset = dataset.map(decode)
epochs = 500
batch_size = 7
batch = dataset.batch(batch_size)
iterator = batch.make_initializable_iterator()
next_element = iterator.get_next()
X, y = next_element




# down path
ld1_1 = tf.layers.batch_normalization(X, training=True)
ld1_2 = tf.layers.batch_normalization(tf.layers.conv2d(ld1_1, 64, [3, 3], padding="valid", activation=tf.nn.relu), training=True)
ld1_3 = tf.layers.batch_normalization(tf.layers.conv2d(ld1_2, 64, [3, 3], padding="valid", activation=tf.nn.relu), training=True)

ld2_1 = tf.layers.max_pooling2d(ld1_3, 2, 2, padding='valid')
ld2_2 = tf.layers.batch_normalization(tf.layers.conv2d(ld2_1, 128, [3, 3], padding="valid", activation=tf.nn.relu), training=True)
ld2_3 = tf.layers.batch_normalization(tf.layers.conv2d(ld2_2, 128, [3, 3], padding="valid", activation=tf.nn.relu), training=True)

ld3_1 = tf.layers.max_pooling2d(ld2_3, 2, 2, padding='valid')
ld3_2 = tf.layers.batch_normalization(tf.layers.conv2d(ld3_1, 256, [3, 3], padding="valid", activation=tf.nn.relu), training=True)
ld3_3 = tf.layers.batch_normalization(tf.layers.conv2d(ld3_2, 256, [3, 3], padding="valid", activation=tf.nn.relu), training=True)

ld4_1 = tf.layers.max_pooling2d(ld3_3, 2, 2, padding='valid')
ld4_2 = tf.layers.batch_normalization(tf.layers.conv2d(ld4_1, 512, [3, 3], padding="valid", activation=tf.nn.relu), training=True)
ld4_3 = tf.layers.batch_normalization(tf.layers.conv2d(ld4_2, 512, [3, 3], padding="valid", activation=tf.nn.relu), training=True)

l5_1 = tf.layers.max_pooling2d(ld4_3, 2, 2, padding='valid')
l5_2 = tf.layers.batch_normalization(tf.layers.conv2d(l5_1, 1024, [3, 3], padding="valid", activation=tf.nn.relu), training=True)
l5_3 = tf.layers.batch_normalization(tf.layers.conv2d(l5_2, 1024, [3, 3], padding="valid", activation=tf.nn.relu), training=True)


# up path
upsamp = tf.layers.batch_normalization(tf.layers.conv2d_transpose(l5_3, 512, [2, 2], strides=[2, 2], padding='valid'), training=True)
copy = ld4_3
begincrop = (int)((copy.shape.as_list()[1] - upsamp.shape.as_list()[1])/2)
sizecrop = upsamp.shape.as_list()[1]
copycrop = slice(copy, [0, begincrop, begincrop, 0], [-1, sizecrop, sizecrop, -1])
lu4_1 = concat([copycrop, upsamp], 3)
lu4_2 = tf.layers.batch_normalization(tf.layers.conv2d(lu4_1, 512, [3, 3], padding="valid", activation=tf.nn.relu), training=True)
lu4_3 = tf.layers.batch_normalization(tf.layers.conv2d(lu4_2, 512, [3, 3], padding="valid", activation=tf.nn.relu), training=True)

upsamp = tf.layers.batch_normalization(tf.layers.conv2d_transpose(lu4_3, 256, [2, 2], strides=[2, 2], padding='valid'), training=True)
copy = ld3_3
begincrop = (int)((copy.shape.as_list()[1] - upsamp.shape.as_list()[1])/2)
sizecrop = upsamp.shape.as_list()[1]
copycrop = slice(copy, [0, begincrop, begincrop, 0], [-1, sizecrop, sizecrop, -1])
lu3_1 = concat([copycrop, upsamp], 3)
lu3_2 = tf.layers.batch_normalization(tf.layers.conv2d(lu3_1, 256, [3, 3], padding="valid", activation=tf.nn.relu), training=True)
lu3_3 = tf.layers.batch_normalization(tf.layers.conv2d(lu3_2, 256, [3, 3], padding="valid", activation=tf.nn.relu), training=True)

upsamp = tf.layers.batch_normalization(tf.layers.conv2d_transpose(lu3_3, 128, [2, 2], strides=[2, 2], padding='valid'), training=True)
copy = ld2_3
begincrop = (int)((copy.shape.as_list()[1] - upsamp.shape.as_list()[1])/2)
sizecrop = upsamp.shape.as_list()[1]
copycrop = slice(copy, [0, begincrop, begincrop, 0], [-1, sizecrop, sizecrop, -1])
lu2_1 = concat([copycrop, upsamp], 3)
lu2_2 = tf.layers.batch_normalization(tf.layers.conv2d(lu2_1, 128, [3, 3], padding="valid", activation=tf.nn.relu), training=True)
lu2_3 = tf.layers.batch_normalization(tf.layers.conv2d(lu2_2, 128, [3, 3], padding="valid", activation=tf.nn.relu), training=True)

upsamp = tf.layers.batch_normalization(tf.layers.conv2d_transpose(lu2_3, 64, [2, 2], strides=[2, 2], padding='valid'), training=True)
copy = ld1_3
begincrop = (int)((copy.shape.as_list()[1] - upsamp.shape.as_list()[1])/2)
sizecrop = upsamp.shape.as_list()[1]
copycrop = slice(copy, [0, begincrop, begincrop, 0], [-1, sizecrop, sizecrop, -1])
lu1_1 = concat([copycrop, upsamp], 3)
lu1_2 = tf.layers.batch_normalization(tf.layers.conv2d(lu1_1, 64, [3, 3], padding="valid", activation=tf.nn.relu), training=True)
lu1_3 = tf.layers.batch_normalization(tf.layers.conv2d(lu1_2, 64, [3, 3], padding="valid", activation=tf.nn.relu), training=True)
lu1_4 = tf.layers.conv2d(lu1_3, 1, [1, 1], padding="valid", activation=tf.sigmoid)

# loss function
cost = tf.reduce_sum(tf.keras.backend.binary_crossentropy(y, lu1_4))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

tf.summary.scalar("cost", cost)
tf.summary.image("trainingImages", X, max_outputs=2)
tf.summary.image("trainingMasks", y, max_outputs=2)
tf.summary.image("trainingPred", lu1_4, max_outputs=2)
summaryOp = tf.summary.merge_all()

config = tf.ConfigProto(device_count={'GPU': 1}, log_device_placement=True)
with tf.Session(config=config) as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    #sess = tf_debug.TensorBoardDebugWrapperSession(sess, "localhost:6066")
    writer = tf.summary.FileWriter("/tmp/log", sess.graph)
    sess.graph.finalize()
    runnr = 1
    for epoch in range(1, epochs+1):
        print("Starting epoch {}...".format(epoch))
        sess.run(iterator.initializer)
        try:
            batchnr = 1
            while True:
                _, c, summary = sess.run([optimizer, cost, summaryOp])
                writer.add_summary(summary, runnr)
                print("Run: {}, Epoch: {}, Batch: {}, Cost: {}".format(runnr, epoch, batchnr, c))
                batchnr += 1
                runnr += 1
        except tf.errors.OutOfRangeError:
            pass
