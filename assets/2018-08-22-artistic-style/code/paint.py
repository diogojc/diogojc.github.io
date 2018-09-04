import tensorflow as tf
import numpy as np
from keras.applications.vgg19 import VGG19
import keras.backend as K


LOG_DIR = "logs"
CONTENT_LAYERS = ["relu4_2"]
STYLE_LAYERS = ["relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"]


def GramMatrix(layer):
    _, h, w, c = layer.get_shape().as_list()
    F = tf.reshape(layer, [h*w, c])
    return tf.matmul(tf.transpose(F), F)


def ContentLoss(F, P):
    return tf.nn.l2_loss(F - P)


def StyleLoss(A, G, M, N):
    return tf.nn.l2_loss(G - A) / (2*N**2*M**2)



# p = content image
# a = style image
# x = generated image

# N = number of channels
# M = height * width

# P = activations of content image
# A = gram matrix of activations of style image

# F = activations of generated image
# G = gram matrix of activations of generated image

# Get weights of trained VGG19
trainedWeights = VGG19(weights='imagenet', include_top=False).get_weights()

# Get layer activations for content image and gram matrices for style image
contentImage = getImageForVGG("content.jpg")
styleImage = getImageForVGG("style.jpg")
with tf.Session() as sess:
    p = tf.constant(contentImage)
    P = {}
    activations = getActivations(p, trainedWeights)
    for l in CONTENT_LAYERS:
        P[l] = activations[l].eval()

    a = tf.constant(styleImage)
    A = {}
    activations = getActivations(a, trainedWeights)
    for l in STYLE_LAYERS:
        A[l] = GramMatrix(activations[l]).eval()
tf.reset_default_graph()
K.clear_session()

# Build NN to optimize generated image
p = tf.constant(contentImage)
for k in P:
    P[k] = tf.constant(P[k])
a = tf.constant(styleImage)
for k in A:
    A[k] = tf.constant(A[k])
x = tf.Variable(getRandomImageForVGG(),
                constraint=lambda xx: tf.clip_by_value(xx, -123, 152))
F = {}
activations = getActivations(x, trainedWeights)
for l in CONTENT_LAYERS:
    F[l] = activations[l]
G = {}
M = {}
N = {}
for l in STYLE_LAYERS:
    G[l] = GramMatrix(activations[l])
    _, h, w, c = activations[l].get_shape().as_list()
    M[l] = h * w
    N[l] = c

preview = tf.concat([x, p, a], axis=2)[..., ::-1]
tf.summary.image("preview", preview, max_outputs=1)

with tf.name_scope("loss"):
    alpha = 10e-5
    beta = 1
    contentLoss = ContentLoss(F["relu4_2"], P["relu4_2"])
    styleLoss = (StyleLoss(A["relu1_1"], G["relu1_1"], M["relu1_1"], N["relu1_1"]) +
                 StyleLoss(A["relu2_1"], G["relu2_1"], M["relu2_1"], N["relu2_1"]) +
                 StyleLoss(A["relu3_1"], G["relu3_1"], M["relu3_1"], N["relu3_1"]) +
                 StyleLoss(A["relu4_1"], G["relu4_1"], M["relu4_1"], N["relu4_1"]) +
                 StyleLoss(A["relu5_1"], G["relu5_1"], M["relu5_1"], N["relu5_1"]))
    styleLoss /= 5
    loss = alpha * contentLoss + beta * styleLoss
    tf.summary.scalar("loss", loss)

optimizer = tf.train.AdamOptimizer(6).minimize(loss, var_list=[x])

config = tf.ConfigProto(device_count={'GPU': 1}, log_device_placement=True)
with tf.Session(config=config) as sess:
    writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
    summaryOp = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())

    for epoch in range(5000):
        _, summary = sess.run([optimizer, summaryOp])
        writer.add_summary(summary, epoch)
