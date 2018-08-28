import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
import numpy as np


def GramMatrix(layer):
    h, w, c = layer.get_shape().as_list()
    F = tf.reshape(layer, [h*w, c])
    return tf.matmul(F, F, transpose_a=True)


def StyleLossForLayer(generatedImageLayer, styleImageLayer):
    h, w, c = generatedImageLayer.get_shape().as_list()
    generatedImageGramMatrix = GramMatrix(generatedImageLayer)
    styleImageGramMatrix = GramMatrix(styleImageLayer)
    tmp = 4*h**2*w**2*c**2
    return tf.reduce_sum(np.square(generatedImageGramMatrix - styleImageGramMatrix)) / tmp


def loadImageToVGG19(path):
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x
content = loadImageToVGG19("content.jpg")[..., ::-1]
style = loadImageToVGG19("style.jpg")[..., ::-1]
_, h, w, c = content.shape
X = np.concatenate((content, style), axis=0).astype(np.float64)
generated = np.random.rand(1, h, w, c)*255


graph = tf.Graph()
with graph.as_default():
    X = tf.convert_to_tensor(X)
    infer = VGG19(weights='imagenet',
                  include_top=False,
                  input_tensor=X)
    infer_conv1_1 = infer.get_layer('block1_conv1').output
    infer_conv2_1 = infer.get_layer('block2_conv1').output
    infer_conv3_1 = infer.get_layer('block3_conv1').output

    g = tf.Variable(generated)
    train = VGG19(weights='imagenet',
                  include_top=False,
                  input_tensor=g)
    train_conv1_1 = train.get_layer('block1_conv1').output
    train_conv2_1 = train.get_layer('block2_conv1').output
    train_conv3_1 = train.get_layer('block3_conv1').output

    #loss = tf.reduce_sum(tf.square(g - X[0])) / 2 + tf.reduce_sum(tf.square(g - X[1])) / 2
    #loss = tf.reduce_sum(tf.square(train_conv4_1[0] - infer_conv4_1[0])) / 2
    #loss = (tf.reduce_sum(tf.square(train_conv1_1[0] - infer_conv1_1[0])) / 2 +
            #StyleLossForLayer(train_conv2_1[0], infer_conv2_1[1]))
    beta = 1
    alpha = 10e-4 * beta
    contentLoss = tf.reduce_sum(tf.square(train_conv3_1[0] - infer_conv3_1[0])) / 2
#     styleCost = (StyleLossForLayer(train_conv1_1[0], infer_conv1_1[1]) +
#                  StyleLossForLayer(train_conv2_1[0], infer_conv2_1[1]) +
#                  StyleLossForLayer(train_conv3_1[0], infer_conv3_1[1]))
    loss = contentLoss
    optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss, var_list=[g])

    config = tf.ConfigProto(device_count={'GPU': 1}, log_device_placement=True)
    with tf.Session(config=config) as sess:
        writer = tf.summary.FileWriter("logs", graph)
        tf.summary.scalar("loss", loss)
        preview = tf.concat([[g[0]], [X[0]], [X[1]]], axis=2)
        tf.summary.image("preview", preview, max_outputs=1)
        summaryOp = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())

        for epoch in range(5000):
            _, l, i, summary = contentLastActivation = sess.run([optimizer, loss, g, summaryOp])
            writer.add_summary(summary, epoch)
