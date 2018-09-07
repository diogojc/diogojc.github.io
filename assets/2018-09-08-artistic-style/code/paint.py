import tensorflow as tf
from functools import reduce
from imageUtils import getImageForVGG, getRandomImageForVGG
from vgg19 import getWeights, getActivations
from artisticStyleUtils import GramMatrix, ContentLoss, StyleLoss


# layers used to calculate content and style losses
CONTENT_LAYERS = ["relu4_2"]
STYLE_LAYERS = ["relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"]

# VGG19 pre-trained network weights
trainedWeights = getWeights()

# content image
h, w = (640, 480)
p = getImageForVGG("content.jpg", h, w)
# style image
a = getImageForVGG("style.jpg", h, w)
# generated image
x = getRandomImageForVGG(h, w)

# number of channels
N = {}
# height * width
M = {}

# activations of content image
P = {}
# gram matrix of activations of style image
A = {}

# activations of generated image
F = {}
# gram matrix of activations of generated image
G = {}



# Calculate i) layer activations for content (P) image and
# ii) gram matrices for style image (A)
with tf.Graph().as_default():
    with tf.Session() as sess:
        activations = getActivations(tf.constant(p), trainedWeights)
        for l in CONTENT_LAYERS:
            P[l] = activations[l].eval()
        activations = getActivations(tf.constant(a), trainedWeights)
        for l in STYLE_LAYERS:
            A[l] = GramMatrix(activations[l]).eval()


# Create trained graph to optimize input based on previous content an style activations
with tf.Graph().as_default():
    x = tf.Variable(x, constraint=lambda xx: tf.clip_by_value(xx, -123, 152))
    for k in P:
        P[k] = tf.constant(P[k])
    for k in A:
        A[k] = tf.constant(A[k])

    activations = getActivations(x, trainedWeights)
    for l in CONTENT_LAYERS:
        F[l] = activations[l]
    with tf.name_scope("generated_grams"):
        for l in STYLE_LAYERS:
            G[l] = GramMatrix(activations[l])
            _, h, w, c = activations[l].get_shape().as_list()
            M[l] = h * w
            N[l] = c

    with tf.name_scope("loss"):
        contentLoss = reduce(tf.add,
                              [ContentLoss(F[l], P[l]) for l in CONTENT_LAYERS])
        styleLoss = reduce(tf.add,
                           [StyleLoss(A[l],
                                      G[l],
                                      M[l],
                                      N[l]) for l in STYLE_LAYERS])
        styleLoss /= len(STYLE_LAYERS)
        alpha = 10e-5
        beta = 5
        loss = alpha * contentLoss + beta * styleLoss

    optimizer = tf.train.AdamOptimizer(5).minimize(loss, var_list=[x])

    config = tf.ConfigProto(device_count={'GPU': 1}, log_device_placement=True)
    with tf.Session(config=config) as sess:
        writer = tf.summary.FileWriter("logs", sess.graph)
        tf.summary.scalar("loss", loss)
        preview = tf.concat([x, p, a], axis=2)[..., ::-1]
        tf.summary.image("preview", preview, max_outputs=1)
        summaryOp = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())

        for epoch in range(5000):
            _, summary = sess.run([optimizer, summaryOp])
            writer.add_summary(summary, epoch)
