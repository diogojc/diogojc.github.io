import tensorflow as tf
from getDatasets import GetDataset, GetEmbeddings

datasetIterator = GetDataset(1000)
X, y = datasetIterator.get_next()

embeddingsIterator = GetEmbeddings()
embeddings = embeddingsIterator.get_next()

Z = tf.nn.embedding_lookup(embeddings, X)

u = 10
rnn_cell = tf.keras.layers.LSTMCell(u)
output, state = tf.keras.layers.RNN(rnn_cell)(Z)

lastOutput = output[:, -1, :]
Wy = tf.get_variable("Wy", shape=[u, 1], dtype=tf.float64)
by = tf.get_variable("by", shape=[1, 1], dtype=tf.float64)
y_hat = tf.sigmoid(tf.matmul(lastOutput, Wy) + by)
loss = tf.losses.mean_squared_error(y, y_hat)
minimizeLossOp = tf.train.AdamOptimizer(0.01).minimize(loss)


with tf.Session() as sess:
    sess.run([tf.global_variables_initializer()])

    epochs = 500
    runnr = 1
    for epoch in range(1, epochs+1):
        sess.run([datasetIterator.initializer,
                  embeddingsIterator.initializer])
        try:
            batchnr = 1
            while True:
                _, c = sess.run([minimizeLossOp, loss])
                print("Run: {}, Epoch: {}, Batch: {}, Cost: {}".format(runnr,
                                                                       epoch,
                                                                       batchnr,
                                                                       c))
                batchnr += 1
                runnr += 1
        except tf.errors.OutOfRangeError:
            pass
