import tensorflow as tf
import numpy as np
from collections import defaultdict


def GetDataset(batchSize):
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

        tokens = sequence_parsed["tokens"]
        sentiment = tf.reshape(context_parsed["sentiment"], [1])

        return tokens, sentiment

    tfrecord = tf.data.TFRecordDataset("dataset.tfrecord")
    dataset = tfrecord.map(decode)
    batch = dataset.padded_batch(batchSize, padded_shapes=([None], [1]))
    iterator = batch.make_initializable_iterator()
    return iterator


def GetEmbeddings():
    def decode(ex):
        features = {
            "embeddings": tf.FixedLenFeature([], dtype=tf.string)
        }
        parsed = tf.parse_single_example(ex, features)
        embedding = tf.decode_raw(parsed["embeddings"], tf.float64)
        embedding.set_shape(200)
        embedding = tf.cast(embedding, tf.float32)
        return embedding

    tfrecord = tf.data.TFRecordDataset("embeddings.tfrecord")
    dataset = tfrecord.map(decode)
    batch = dataset.batch(400001)
    iterator = batch.make_initializable_iterator()
    return iterator


def testGetDataset():
    tf.reset_default_graph()
    iterator = GetDataset(10)
    X, y = iterator.get_next()
    print(X)

    with tf.Session() as sess:
        sess.run([iterator.initializer])
        for i in range(2):
            try:
                while True:
                    a, b = sess.run([X, y])
                    print(a.shape)
            except tf.errors.OutOfRangeError:
                pass


def TestEmbeddings():
    tf.reset_default_graph()
    iterator = GetEmbeddings()
    Z = iterator.get_next()
    print(Z)

    with tf.Session() as sess:
        sess.run([iterator.initializer])
        a = sess.run([Z])[0]
        print(a.shape)
        print(a[0])


def TestDatasetEmbeddings():
    datasetIterator = GetDataset(10)
    X, y = datasetIterator.get_next()

    embeddingsIterator = GetEmbeddings()
    embeddings = embeddingsIterator.get_next()

    Z = tf.nn.embedding_lookup(embeddings, X)

    print(X)
    print(embeddings)
    print(Z)

    with tf.Session() as sess:
        sess.run([datasetIterator.initializer, embeddingsIterator.initializer])
        a = sess.run([Z])[0]
        print(a.shape)


word_to_index = None


def ConvertToDataset(lines):
    global word_to_index
    if word_to_index is None:
        word_to_index = dict()

        glove_filename = "glove.6B.200d.txt"
        with open(glove_filename, 'r', encoding="utf-8") as glove_file:
            for (i, line) in enumerate(glove_file):
                split = line.split(' ')
                word = split[0]
                word_to_index[word] = i
        word_to_index = defaultdict(lambda: word_to_index["unk"], word_to_index)

    def normalizeWord(w):
        if None:
            raise Exception("word cannot be None")
        return ''.join(w.split()).lower()

    X = [[word_to_index[normalizeWord(w)]
          for w in line.split(" ")]
         for line in lines]
    return np.array(X)