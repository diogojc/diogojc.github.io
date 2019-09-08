import tensorflow as tf
import numpy as np
from Dataset import GetDataset
from Embeddings import GetEmbeddings
from datetime import datetime

trainDataset, testDataset = GetDataset(10000)
handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(handle,
                                               trainDataset.output_types,
                                               trainDataset.output_shapes)
text, label, tokens = iterator.get_next()

trainIterator = trainDataset.make_initializable_iterator()
testIterator = testDataset.make_initializable_iterator()

embeddings = GetEmbeddings()
vocabSize, numberOfEmbeddings = embeddings.shape
embeddedTokens = tf.keras.layers.Embedding(vocabSize+1,
                                           numberOfEmbeddings,
                                           embeddings_initializer=tf.keras.initializers.Constant(embeddings),
                                           mask_zero=True,
                                           trainable=False)(tokens)

cell = tf.keras.layers.LSTMCell(50)
rnn = tf.keras.layers.RNN(cell)
semantics = rnn(embeddedTokens)

prediction = tf.keras.layers.Dense(1, activation='sigmoid')(semantics)

lossOp = tf.losses.mean_squared_error(label, prediction)
minimizeLossOp = tf.train.AdamOptimizer(0.005).minimize(lossOp)

tf.summary.histogram("embeddings", embeddings)
tf.summary.histogram("embeddedTokens", embeddedTokens)
tf.summary.histogram("semantics", semantics)
tf.summary.scalar("loss", lossOp)

predictionsViz = tf.concat([text[:10],
                            tf.as_string(label[:10]),
                            tf.as_string(prediction[:10])],
                            axis=1)
tf.summary.text("predictions", predictionsViz)
summaryOp = tf.summary.merge_all()

with tf.Session() as sess:
    now = datetime.now().isoformat()
    writerTrain = tf.summary.FileWriter("logs/{}/train".format(now), sess.graph)
    writerTest = tf.summary.FileWriter("logs/{}/test".format(now), sess.graph)
    sess.run([tf.global_variables_initializer(),
              tf.local_variables_initializer()])
    epochs = 10
    runnr = 1

    trainHandle = sess.run(trainIterator.string_handle())
    testHandle = sess.run(testIterator.string_handle())

    for epoch in range(1, epochs+1):
        sess.run([trainIterator.initializer,
                  testIterator.initializer])
        try:
            batchnr = 1
            while True:
                # Train
                _, lossTrain, summaryTrain = sess.run([minimizeLossOp,
                                                      lossOp,
                                                      summaryOp],
                                                      feed_dict={
                                                        handle: trainHandle
                                                      })
                writerTrain.add_summary(summaryTrain, global_step=runnr)
                writerTrain.flush()

                # Test
                lossTest, summaryTest = sess.run([lossOp,
                                                  summaryOp],
                                                  feed_dict={
                                                      handle: testHandle
                                                  })
                writerTest.add_summary(summaryTest, global_step=runnr)
                writerTest.flush()

                # print stats and iterate
                output = "Run: {}, Epoch: {}, Batch: {}, Loss: (Train){}  (Test){}"
                print(output.format(runnr,
                                    epoch,
                                    batchnr,
                                    lossTrain,
                                    lossTest))
                batchnr += 1
                runnr += 1
        except tf.errors.OutOfRangeError:
            pass