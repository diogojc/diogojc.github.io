import tensorflow as tf
from getDatasets import GetDataset


def LSTMCell(units, Xt, prev_activation=None, prev_state=None):
    if prev_activation is None:
        prev_activation = tf.zeros([1, units], tf.int32)
    if prev_state is None:
        prev_state = tf.zeros([1, units], tf.int32)

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


def LSTMNetwork(cell, X):
    _, T, _ = X.get_shape().as_list()
    with tf.variable_scope("LSTMNetwork", reuse=True):
        units = 30
        activation = None
        state = None
        activations = []
        for t in range(T):
            activation, state = cell(units, X[:, t, :], activation, state)
            activations.append(activation)
    return activations


T = 10  # number of timesteps for training
n = 1000  # vocab size
# iterator = GetDataset()
# X, y = iterator.get_next()
X = tf.placeholder(tf.int32, [None, T, n])
y = tf.placeholder(tf.int32, [None, 1])

activations = LSTMNetwork(LSTMCell, X)
lastActivation = activations[-1]
Wy = tf.get_variable("Wy", shape=[u, 1])
by = tf.get_variable("by", shape=[1, 1])
y_hat = tf.sigmoid(tf.matmul(lastActivation, Wy) + by)
loss = tf.losses.mean_squared_error(y, y_hat)

minimizeLoss = tf.train.AdamOptimizer(5).minimize(loss)

config = tf.ConfigProto(device_count={'GPU': 1}, log_device_placement=True)
with tf.Session(config=config) as sess:
    writer = tf.summary.FileWriter("logs", sess.graph)
    tf.summary.scalar("loss", loss)
    summaryOp = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())

    for epoch in range(5000):
        _, summary = sess.run([minimizeLoss, summaryOp])
        writer.add_summary(summary, epoch)
