import tensorflow as tf
from collections import defaultdict
import numpy as np
import pdb


def init():
    word_to_index = dict()
    embeddings = []

    glove_filename = "glove.6B.200d.txt"
    with open(glove_filename, 'r', encoding="utf-8") as glove_file:
        for (i, line) in enumerate(glove_file):
            split = line.split(' ')
            word = split[0]
            representation = split[1:]
            representation = [float(val) for val in representation]

            word_to_index[word] = i
            embeddings.append(representation)

    _WORD_NOT_FOUND = [0.0] * len(representation)
    word_to_index = defaultdict(lambda: word_to_index["unk"], word_to_index)
    embeddings = np.array(embeddings + [_WORD_NOT_FOUND])
    return word_to_index, embeddings


def createSentimentDataset(word_to_index):
    def normalizeWord(w):
        if None:
            raise Exception("word cannot be None")
        return ''.join(w.split()).lower()

    y = dict()
    with open("sentiment_labels.txt", 'r', encoding="utf-8") as y_file:
        next(y_file)  # skip header
        for line in y_file:
            split = line.split("|")
            y[split[0]] = float(split[1])

    with open("dictionary.txt", 'r', encoding="utf-8") as X_file:
        writer = tf.python_io.TFRecordWriter("dataset.tfrecord")
        for line in X_file:
            split = line.rstrip('\n').split("|")
            wordIds = [word_to_index[normalizeWord(w)] for w in split[0].split(" ")]
            sentiment = y[split[1]]

            c = tf.train.Features(feature={
                "sentiment": tf.train.Feature(float_list=tf.train.FloatList(value=[sentiment])),
            })

            fl = []
            for wid in wordIds:
                tmp = tf.train.Feature(int64_list=tf.train.Int64List(
                            value=[wid]
                        ))
                fl.append(tmp)

            fls = tf.train.FeatureLists(feature_list={
                "tokens": tf.train.FeatureList(feature=fl)
            })
            se = tf.train.SequenceExample(context=c, feature_lists=fls)

            writer.write(se.SerializeToString())
        writer.close()


def createEmbeddingsDataset(embeddings):
    writer = tf.python_io.TFRecordWriter("embeddings.tfrecord")
    for i in range(embeddings.shape[0]):
        features = {
            "embeddings": tf.train.Feature(bytes_list=tf.train.BytesList(value=[embeddings[i, :].tostring()])),
        }
        example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(example.SerializeToString())
    writer.close()


wti, embeddings = init()
createSentimentDataset(wti)
createEmbeddingsDataset(embeddings)
