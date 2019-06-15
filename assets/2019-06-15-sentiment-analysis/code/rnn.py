import tensorflow as tf
from getDatasets import GetDataset, GetEmbeddings, ConvertToDataset
import numpy as np

with tf.Session() as sess:
    trainDatasetIterator = GetDataset(1000)
    train_iterator_handle = sess.run(trainDatasetIterator.string_handle())

    testData = ConvertToDataset(["I didn't like it."])
    print(testData)
    testData = (testData, [[.0]])
    testDatasetIterator = tf.data.Dataset.from_tensor_slices(testData).make_one_shot_iterator()
    test_iterator_handle = sess.run(testDatasetIterator.string_handle())

    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle,
                                                   trainDatasetIterator.output_types,
                                                   output_shapes=trainDatasetIterator.output_shapes)

    embeddingsIterator = GetEmbeddings()
    embeddings = embeddingsIterator.get_next()

    X, y = iterator.get_next()

    Z = tf.keras.backend.cast(tf.nn.embedding_lookup(embeddings, X), dtype="float32")

    u = 10
    print(Z)
    rnn_cell = tf.keras.layers.LSTMCell(u)
    output, state = tf.keras.layers.RNN(rnn_cell)(Z)

    lastOutput = output[:, -1, :]
    Wy = tf.get_variable("Wy", shape=[u, 1], dtype=tf.float64)
    by = tf.get_variable("by", shape=[1, 1], dtype=tf.float64)
    y_hat = tf.sigmoid(tf.matmul(lastOutput, Wy) + by)

    loss = tf.losses.mean_squared_error(y, y_hat)
    minimizeLossOp = tf.train.AdamOptimizer(0.01).minimize(loss)

    tf.summary.scalar("loss", loss)
    summaryOp = tf.summary.merge_all()
    sess.run([tf.global_variables_initializer()])
    writer = tf.summary.FileWriter("logs", sess.graph)
    epochs = 2
    runnr = 1
    for epoch in range(1, epochs+1):
        sess.run([trainDatasetIterator.initializer,
                  embeddingsIterator.initializer])
        try:
            batchnr = 1
            while True:
                _, c, summary = sess.run([minimizeLossOp, loss, summaryOp],
                                         feed_dict={handle: train_iterator_handle})
                writer.add_summary(summary, runnr)
                print("Run: {}, Epoch: {}, Batch: {}, Cost: {}".format(runnr,
                                                                       epoch,
                                                                       batchnr,
                                                                       c))
                batchnr += 1
                runnr += 1
        except tf.errors.OutOfRangeError:
            pass


    sess.run([embeddingsIterator.initializer])
    a = sess.run([y_hat], feed_dict={handle: test_iterator_handle})
    print(a)
