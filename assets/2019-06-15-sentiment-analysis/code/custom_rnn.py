import tensorflow as tf
from getDatasets import GetDataset, GetEmbeddings


class MyLSTMCell():

    def build():
       # define your own logic

    def call():
      # call your own logic


class MyLSTMCell(tf.keras.layers.Layer):
    def __init__(self, units):
        self._num_units = units

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def build(self, input_shape):
        raise NotImplementedError

    def call(X)

    def call(Xt, prev_activation=None, prev_state=None):
        units = self._num_units

        if prev_activation is None:
            prev_activation = tf.zeros([1, units], tf.float32)
        if prev_state is None:
            prev_state = tf.zeros([1, units], tf.float32)

        _, n = Xt.get_shape().as_list()
        _, u = prev_activation.get_shape().as_list()

        assert units == u

        concat = tf.concat([prev_activation, Xt], axis=1)

        Wf = tf.get_variable("Wf", shape=[n+u, u])
        bf = tf.get_variable("bf", shape=[1, u])
        ft = tf.sigmoid(tf.matmul(concat, Wf) + bf)

        Wi = tf.get_variable("Wi", shape=[n+u, u])
        bi = tf.get_variable("bi", shape=[1, u])
        it = tf.sigmoid(tf.matmul(Wi, concat) + bi)

        Wc = tf.get_variable("Wc", shape=[n+u, u])
        bc = tf.get_variable("bc", shape=[1, u])
        ct = tf.tanh(tf.matmul(Wc, concat) + bc)

        curr_state = ft * prev_state + it * ct

        Wo = tf.get_variable("Wo", shape=[n+u, u])
        bo = tf.get_variable("bo", shape=[1, u])
        ot = tf.sigmoid(tf.matmul(Wo, concat) + bo)

        curr_activation = ot * tf.tanh(curr_state)

        return (curr_activation, curr_state)

# def LSTMNetwork(cell, X):
#     _, T, _ = X.get_shape().as_list()
#     with tf.variable_scope("LSTMNetwork", reuse=True):
#         units = 30
#         activation = None
#         state = None
#         activations = []
#         for t in range(T):
#             activation, state = cell(units, X[:, t, :], activation, state)
#             activations.append(activation)
#     return activations


datasetIterator = GetDataset(1000)
X, y = datasetIterator.get_next()

embeddingsIterator = GetEmbeddings()
embeddings = embeddingsIterator.get_next()

Z = tf.nn.embedding_lookup(embeddings, X)

u = 10
rnn_cell = tf.nn.rnn_cell.LSTMCell(u)

output, state = tf.nn.dynamic_rnn(rnn_cell,
                                  Z,
                                  dtype=tf.float64)

# activations = LSTMNetwork(LSTMUnit, Z)
# lastActivation = activations[-1]

lastOutput = output[:, -1, :]
Wy = tf.get_variable("Wy", shape=[u, 1], dtype=tf.float64)
by = tf.get_variable("by", shape=[1, 1], dtype=tf.float64)
y_hat = tf.sigmoid(tf.matmul(lastOutput, Wy) + by)
loss = tf.losses.mean_squared_error(y, y_hat)
minimizeLossOp = tf.train.AdamOptimizer(0.01).minimize(loss)

tf.summary.scalar("loss", loss)
summaryOp = tf.summary.merge_all()


config = tf.ConfigProto(device_count={'GPU': 1}, log_device_placement=True)
with tf.Session(config=config) as sess:
    sess.run([tf.global_variables_initializer()])
    writer = tf.summary.FileWriter("logs", sess.graph)
    epochs = 500
    runnr = 1
    for epoch in range(1, epochs+1):
        sess.run([datasetIterator.initializer,
                  embeddingsIterator.initializer])
        try:
            batchnr = 1
            while True:
                _, c, summary = sess.run([minimizeLossOp, loss, summaryOp])
                writer.add_summary(summary, runnr)
                print("Run: {}, Epoch: {}, Batch: {}, Cost: {}".format(runnr,
                                                                       epoch,
                                                                       batchnr,
                                                                       c))
                batchnr += 1
                runnr += 1
        except tf.errors.OutOfRangeError:
            pass
