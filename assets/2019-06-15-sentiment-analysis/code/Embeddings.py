import tensorflow as tf
import numpy as np

glove_filename = "glove.6B.200d.txt"
fileNameEmbeddings = "embeddings.npy"


def GetEmbeddings():
    try:
        with open(fileNameEmbeddings, 'r'):
            pass
    except FileNotFoundError:
        # parse glove file
        embeddings = []
        with open(glove_filename, 'r', encoding="utf-8") as glove_file:
            for line in glove_file:
                split = line.split(' ')
                representation = split[1:]
                representation = [float(val) for val in representation]
                embeddings.append(representation)

        avg_embedding = np.average(embeddings, axis=0)
        pad_embedding = np.zeros(len(embeddings[0]))
        embeddings = np.array([pad_embedding] + embeddings + [avg_embedding])
        np.save(fileNameEmbeddings, embeddings)
        return embeddings
    embeddings = np.load(fileNameEmbeddings)
    return embeddings

def _testEmbeddings():
    z = GetEmbeddings()
    print("{}".format(z.shape))

#_testEmbeddings()