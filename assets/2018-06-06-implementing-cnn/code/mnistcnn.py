import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def CreateConvLayer(previousLayer, filterCount, filterSize):
    return tf.layers.conv2d(previousLayer, filterCount, filterSize, padding="valid", activation=tf.nn.relu)


def CreatePoolingLayer(previousLayer, pool_size):
    return tf.layers.max_pooling2d(previousLayer, pool_size, pool_size)


def CreateFCNLayer(previousLayer, num_outputs, activation=tf.nn.relu):
    return tf.layers.dense(previousLayer, num_outputs, activation=activation)


def CalculateFilterSizeFromOutput(inputSize, outputSize, padding=0, stride=1):
    return -((outputSize-1)*stride-inputSize-2*padding)


X = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
y = tf.placeholder(tf.float32, shape=(None, 10))

l0 = tf.pad(X, [[0, 0], [2, 2], [2, 2], [0, 0]])
l1 = CreateConvLayer(l0, 6, CalculateFilterSizeFromOutput(32, 28))
l2 = CreatePoolingLayer(l1, 2)
l3 = CreateConvLayer(l2, 16, CalculateFilterSizeFromOutput(14, 10))
l4 = CreatePoolingLayer(l3, 2)
l5 = tf.layers.Flatten()(l4)
l6 = CreateFCNLayer(l5, 120)
l7 = tf.layers.dropout(l6, 0.4)
l8 = CreateFCNLayer(l7, 84)
l9 = tf.layers.dropout(l8, 0.4)
l10 = CreateFCNLayer(l9, 10, None)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=l10, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
accuracy = tf.metrics.accuracy(tf.argmax(y, 1), tf.argmax(l10, 1))

with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    # train_writer = tf.summary.FileWriter('./logs/train ', sess.graph)
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    for step in range(20000):
        train = mnist.train.next_batch(100)
        X_train = train[0].reshape(-1, 28, 28, 1)
        y_train = train[1]
        _, c, acc = sess.run([optimizer, cost, accuracy], feed_dict={X: X_train, y: y_train})

        test = mnist.test.next_batch(50)
        X_test = test[0].reshape(-1, 28, 28, 1)
        y_test = test[1]
        acc2 = sess.run([accuracy], feed_dict={X: X_test, y: y_test})
        if step % 2000:
            print("[Step %s] Cost: %s Accuracy (Train): %s Accuracy (Test): %s" % (step, c, acc[0], acc2[0]))
