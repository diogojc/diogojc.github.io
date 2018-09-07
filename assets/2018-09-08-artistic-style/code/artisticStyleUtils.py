import tensorflow as tf


def GramMatrix(layer):
    _, h, w, c = layer.get_shape().as_list()
    F = tf.reshape(layer, [h*w, c])
    return tf.matmul(tf.transpose(F), F)


def ContentLoss(F, P):
    return tf.nn.l2_loss(F - P)


def StyleLoss(A, G, M, N):
    return tf.nn.l2_loss(G - A) / (2*N**2*M**2)