import tensorflow as tf
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
import numpy as np

Wn = 2
X = tf.placeholder(tf.float32, shape=(None, Wn))
y = tf.placeholder(tf.float32, shape=(None, 1))
W = tf.Variable(tf.random_normal([1, Wn]), dtype=tf.float32)
b = tf.Variable(tf.zeros([1, 1]), dtype=tf.float32)
z = tf.add(tf.matmul(X, tf.transpose(W)), b)  # linear combination with bias
a = 1 / (1 + tf.exp(-z))  # sigmoid function
cost = tf.losses.log_loss(y, a)


X_data, y_data = make_blobs(n_samples=5000, centers=2, n_features=Wn)
m = X_data.shape[0]
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(1000):
        sess.run(optimizer, feed_dict={X: X_data, y: y_data.reshape(m, 1)})


    x_min, x_max = X_data[:, 0].min() - 1, X_data[:, 0].max() + 1
    y_min, y_max = X_data[:, 1].min() - 1, X_data[:, 1].max() + 1
    h = .02
    x1, x2 = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    nm = x1.shape[0] * x1.shape[1]
    X_newdata = np.concatenate((x1.reshape((nm, 1)), x2.reshape((nm, 1))), axis=1)
    y_newdata = sess.run(a, feed_dict={X: X_newdata}) > 0.5
    fig, ax = plt.subplots()
    ax.contourf(x1, x2, y_newdata.reshape(x1.shape), cmap=plt.cm.Paired)
    ax.scatter(X_data[:, 0], X_data[:, 1], c=y_data, edgecolors='w', cmap=plt.cm.Paired)
    plt.show()
