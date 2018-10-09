import tensorflow as tf


def GetDataset():
    def decode(ex):
        context_features = {
            "sentiment": tf.FixedLenFeature([], dtype=tf.float32)
        }
        sequence_features = {
            "tokens": tf.FixedLenSequenceFeature([], dtype=tf.int64)
        }

        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized=ex,
            context_features=context_features,
            sequence_features=sequence_features
        )

        return sequence_parsed["tokens"], context_parsed["sentiment"]

    tfrecord = tf.data.TFRecordDataset("dataset.tfrecord")
    dataset = tfrecord.map(decode)
    batch = dataset.padded_batch(10, padded_shapes=([None], []))
    iterator = batch.make_initializable_iterator()
    return iterator


tf.reset_default_graph()
iterator = GetDataset()
X, y = iterator.get_next()

with tf.Session() as sess:
    sess.run([iterator.initializer])
    for i in range(10):
        a, b = sess.run([X, y])
        print(a[0, :])
        print(a.shape)
        print(b.shape)
