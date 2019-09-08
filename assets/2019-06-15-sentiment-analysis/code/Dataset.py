import tensorflow as tf
from collections import defaultdict
import random

fileNameDataset = "dataset2.tfrecord"
trainFileName = "train.tfrecord"
testFileName = "test.tfrecord"
glove_filename = "glove.6B.200d.txt"

def _createDataset(trainPercentage):
    y = dict()
    with open("sentiment_labels.txt", 'r', encoding="utf-8") as y_file:
        next(y_file)  # skip file header
        for line in y_file:
            split = line.split("|")
            y[split[0]] = float(split[1])

    with open("dictionary.txt", 'r', encoding="utf-8") as X_file:
        # build vocabulary from glove
        word_to_index = dict()
        with open(glove_filename, 'r', encoding="utf-8") as glove_file:
            for (i, line) in enumerate(glove_file):
                split = line.split(' ')
                word = split[0]
                word_to_index[word] = i+1
        word_to_index = defaultdict(lambda: i+2, word_to_index)

        # build dataset with tokenized representation using glove vocab
        dataset = []
        for line in X_file:
            split = line.rstrip('\n').split("|")
            
            text = split[0].encode("utf-8")
            sentiment = y[split[1]]
            wordIds = [word_to_index[w] for w in split[0].split(" ")]
            dataset.append((text, wordIds, sentiment))

        random.seed(12345)
        random.shuffle(dataset)

        def writeTFRecord(dataset, fileName):
            writer = tf.python_io.TFRecordWriter(fileName)
            for text, wordIds, sentiment in dataset: 
                c = tf.train.Features(feature={
                    "sentiment": tf.train.Feature(float_list=tf.train.FloatList(value=[sentiment])),
                    "text": tf.train.Feature(bytes_list=tf.train.BytesList(value=[text]))
                })
                
                fl = []
                for wid in wordIds:
                    tmp = tf.train.Feature(int64_list=tf.train.Int64List(value=[wid]))
                    fl.append(tmp)
                fls = tf.train.FeatureLists(feature_list={
                    "tokens": tf.train.FeatureList(feature=fl)
                })

                se = tf.train.SequenceExample(context=c, feature_lists=fls)

                writer.write(se.SerializeToString())
            writer.close()

        cutoff = int(len(dataset)*trainPercentage)
        writeTFRecord(dataset[:cutoff], trainFileName)
        writeTFRecord(dataset[cutoff:], testFileName)


def GetDataset(batchSize, split=0.7):
    try:
        with open(trainFileName, 'r'):
            pass
        with open(testFileName, 'r'):
            pass
    except FileNotFoundError:
        _createDataset(split)

    def decode(ex):
        context_features = {
            "sentiment": tf.FixedLenFeature([], dtype=tf.float32),
            "text": tf.FixedLenFeature([], dtype=tf.string),
        }
        sequence_features = {
            "tokens": tf.FixedLenSequenceFeature([], dtype=tf.int64)
        }

        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized=ex,
            context_features=context_features,
            sequence_features=sequence_features
        )

        sentiment = tf.reshape(context_parsed["sentiment"], [1])
        text = tf.reshape(context_parsed["text"], [1])
        tokens = sequence_parsed["tokens"]

        return (text, sentiment, tokens)

    trainTFRecord = tf.data.TFRecordDataset(trainFileName)
    testTFRecord = tf.data.TFRecordDataset(testFileName)
    train = trainTFRecord.map(decode)
    test = testTFRecord.map(decode)
    trainBatched = train.padded_batch(batchSize, padded_shapes=([1], [1], [None]))
    testBatched = test.padded_batch(batchSize, padded_shapes=([1], [1], [None]))
    return (trainBatched, testBatched)


def _testDataset():
    train, _ = GetDataset(1)
    iterator = train.make_initializable_iterator()
    T, y, X = iterator.get_next()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer)
        try:
            while True:
                text, label, tokens = sess.run([T, y, X])
                print("{} - {} - {}".format(label, text, tokens))
        except tf.errors.OutOfRangeError:
            pass

#_testDataset()