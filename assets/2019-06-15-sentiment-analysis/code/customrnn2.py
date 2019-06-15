import tensorflow as tf
import numpy as np
from getDatasets import GetDataset, GetEmbeddings, ConvertToDataset

datasetIterator = GetDataset(25000)
X, y = datasetIterator.get_next()

embeddingsIterator = GetEmbeddings()
embeddings = embeddingsIterator.get_next()

Z = tf.nn.embedding_lookup(embeddings, X)

units = 25
cell = tf.keras.layers.LSTMCell(units)
net = tf.keras.layers.RNN(cell)
lastOutput = net(Z)

Wy = tf.get_variable("Wy", shape=[units, 1], dtype=tf.float32)
by = tf.get_variable("by", shape=[1, 1], dtype=tf.float32)
y_hat = tf.sigmoid(tf.matmul(lastOutput, Wy) + by)

lossOp = tf.losses.sigmoid_cross_entropy(y, y_hat)
minimizeLossOp = tf.train.AdamOptimizer(0.01).minimize(lossOp)

_, aucOp = tf.metrics.auc(y, y_hat)

tf.summary.scalar("AUC", aucOp)
tf.summary.scalar("loss", lossOp)
summaryOp = tf.summary.merge_all()

with tf.Session() as sess:
    writer = tf.summary.FileWriter("logs", sess.graph)
    sess.run([tf.global_variables_initializer(),
              tf.local_variables_initializer()])
    epochs = 5
    runnr = 1
    for epoch in range(1, epochs+1):
        sess.run([datasetIterator.initializer,
                  embeddingsIterator.initializer])
        try:
            batchnr = 1
            while True:
                _, loss, summary = sess.run([minimizeLossOp, lossOp, summaryOp])
                writer.add_summary(summary, runnr)
                print("Run: {}, Epoch: {}, Batch: {}, Loss: {}".format(runnr,
                                                                       epoch,
                                                                       batchnr,
                                                                       loss))
                batchnr += 1
                runnr += 1
                sess.run([embeddingsIterator.initializer])
        except tf.errors.OutOfRangeError:
            pass
