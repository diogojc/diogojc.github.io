import tensorflow as tf
import numpy as np
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.applications.vgg19 import VGG19
import keras.backend as K
from keras.models import Model

def GramMatrix(layer):
    _, h, w, c = layer.get_shape().as_list()
    F = tf.reshape(layer, [h*w, c])
    return tf.matmul(tf.transpose(F), F)


def StyleLossForLayer(generatedImageLayer, styleImageLayer):
    _, h, w, c = generatedImageLayer.get_shape().as_list()
    generatedImageGramMatrix = GramMatrix(generatedImageLayer)
    styleImageGramMatrix = GramMatrix(styleImageLayer)
    tmp = 2*c**2*(h*w)**2
    return tf.nn.l2_loss(generatedImageGramMatrix - styleImageGramMatrix) / tmp





def getImage(path):
    img = image.load_img(path, target_size=(512, 512))
    arr = image.img_to_array(img)
    arre = np.expand_dims(arr, axis=0)
    pp = preprocess_input(arre)
    return pp

MEAN_VALUE = np.array([103.939, 116.779, 123.68])   # BGR
def getRandomTensorForVGG():
    arre = (np.random.rand(1, 512, 512, 3)*255),
    pp = arre[0] - MEAN_VALUE
    # https://stackoverflow.com/questions/46545986/how-to-use-tf-clip-by-value-on-sliced-tensor-in-tensorflow
    return tf.Variable(pp.astype(np.float32), constraint=lambda x: tf.clip_by_value(x, -123, 152))


def addNoise(image, sigma=30):
    m, h, w, c = image.shape
    noise = np.random.normal(loc=sigma/2, scale=sigma, size=[m, h, w, c])
    imageWithNoise = image + noise
    return np.clip(imageWithNoise, 0, 255).astype(np.float32)



trainedWeights = VGG19(weights='imagenet', include_top=False).get_weights()
tf.reset_default_graph()
K.clear_session()


def getModel(X, trainedWeights):
    with tf.name_scope("conv1_1"):
        conv1_1_W = tf.constant(trainedWeights[0], name="W")
        conv1_1_b = tf.constant(trainedWeights[1], name="b")
        conv1_1 = tf.nn.bias_add(tf.nn.conv2d(X, conv1_1_W, strides=[1, 1, 1, 1], padding='SAME'),
                                 conv1_1_b)
        relu1_1 = tf.nn.relu(conv1_1)
        tf.summary.histogram("weights", conv1_1_W)
        tf.summary.histogram("bias", conv1_1_b)
        tf.summary.histogram("relu", relu1_1)

    with tf.name_scope("conv1_2"):
        conv1_2_W = tf.constant(trainedWeights[2], name="W")
        conv1_2_b = tf.constant(trainedWeights[3], name="b")
        conv1_2 = tf.nn.bias_add(tf.nn.conv2d(relu1_1, conv1_2_W, strides=[1, 1, 1, 1], padding='SAME'),
                                 conv1_2_b)
        relu1_2 = tf.nn.relu(conv1_2)
        tf.summary.histogram("weights", conv1_2_W)
        tf.summary.histogram("bias", conv1_2_b)
        tf.summary.histogram("relu", relu1_2)

    with tf.name_scope("pool1"):
        pool1 = tf.nn.pool(relu1_2, [2, 2], "AVG", "VALID", strides=[2, 2])

    with tf.name_scope("conv2_1"):
        conv2_1_W = tf.constant(trainedWeights[4], name="W")
        conv2_1_b = tf.constant(trainedWeights[5], name="b")
        conv2_1 = tf.nn.bias_add(tf.nn.conv2d(pool1, conv2_1_W, strides=[1, 1, 1, 1], padding='SAME'),
                                 conv2_1_b)
        relu2_1 = tf.nn.relu(conv2_1)
        tf.summary.histogram("weights", conv2_1_W)
        tf.summary.histogram("bias", conv2_1_b)
        tf.summary.histogram("relu", relu2_1)

    with tf.name_scope("conv2_2"):
        conv2_2_W = tf.constant(trainedWeights[6], name="W")
        conv2_2_b = tf.constant(trainedWeights[7], name="b")
        conv2_2 = tf.nn.bias_add(tf.nn.conv2d(relu2_1, conv2_2_W, strides=[1, 1, 1, 1], padding='SAME'),
                                 conv2_2_b)
        relu2_2 = tf.nn.relu(conv2_2)
        tf.summary.histogram("weights", conv2_2_W)
        tf.summary.histogram("bias", conv2_2_b)
        tf.summary.histogram("relu", relu2_2)

    with tf.name_scope("pool2"):
        pool2 = tf.nn.pool(relu2_2, [2, 2], "AVG", "VALID", strides=[2, 2])

    with tf.name_scope("conv3_1"):
        conv3_1_W = tf.constant(trainedWeights[8], name="W")
        conv3_1_b = tf.constant(trainedWeights[9], name="b")
        conv3_1 = tf.nn.bias_add(tf.nn.conv2d(pool2, conv3_1_W, strides=[1, 1, 1, 1], padding='SAME'),
                                 conv3_1_b)
        relu3_1 = tf.nn.relu(conv3_1)
        tf.summary.histogram("weights", conv3_1_W)
        tf.summary.histogram("bias", conv3_1_b)
        tf.summary.histogram("relu", relu3_1)

    with tf.name_scope("conv3_2"):
        conv3_2_W = tf.constant(trainedWeights[10], name="W")
        conv3_2_b = tf.constant(trainedWeights[11], name="b")
        conv3_2 = tf.nn.bias_add(tf.nn.conv2d(relu3_1, conv3_2_W, strides=[1, 1, 1, 1], padding='SAME'),
                                 conv3_2_b)
        relu3_2 = tf.nn.relu(conv3_2)
        tf.summary.histogram("weights", conv3_2_W)
        tf.summary.histogram("bias", conv3_2_b)
        tf.summary.histogram("relu", relu3_2)

    with tf.name_scope("conv3_3"):
        conv3_3_W = tf.constant(trainedWeights[12], name="W")
        conv3_3_b = tf.constant(trainedWeights[13], name="b")
        conv3_3 = tf.nn.bias_add(tf.nn.conv2d(relu3_2, conv3_3_W, strides=[1, 1, 1, 1], padding='SAME'),
                                 conv3_3_b)
        relu3_3 = tf.nn.relu(conv3_3)
        tf.summary.histogram("weights", conv3_3_W)
        tf.summary.histogram("bias", conv3_3_b)
        tf.summary.histogram("relu", relu3_3)

    with tf.name_scope("conv3_4"):
        conv3_4_W = tf.constant(trainedWeights[14], name="W")
        conv3_4_b = tf.constant(trainedWeights[15], name="b")
        conv3_4 = tf.nn.bias_add(tf.nn.conv2d(relu3_3, conv3_4_W, strides=[1, 1, 1, 1], padding='SAME'),
                                 conv3_4_b)
        relu3_4 = tf.nn.relu(conv3_4)
        tf.summary.histogram("weights", conv3_4_W)
        tf.summary.histogram("bias", conv3_4_b)
        tf.summary.histogram("relu", relu3_4)

    with tf.name_scope("pool3"):
        pool3 = tf.nn.pool(relu3_4, [2, 2], "AVG", "VALID", strides=[2, 2])

    with tf.name_scope("conv4_1"):
        conv4_1_W = tf.constant(trainedWeights[16], name="W")
        conv4_1_b = tf.constant(trainedWeights[17], name="b")
        conv4_1 = tf.nn.bias_add(tf.nn.conv2d(pool3, conv4_1_W, strides=[1, 1, 1, 1], padding='SAME'),
                                 conv4_1_b)
        relu4_1 = tf.nn.relu(conv4_1)
        tf.summary.histogram("weights", conv4_1_W)
        tf.summary.histogram("bias", conv4_1_b)
        tf.summary.histogram("relu", relu4_1)

    with tf.name_scope("conv4_2"):
        conv4_2_W = tf.constant(trainedWeights[18], name="W")
        conv4_2_b = tf.constant(trainedWeights[19], name="b")
        conv4_2 = tf.nn.bias_add(tf.nn.conv2d(relu4_1, conv4_2_W, strides=[1, 1, 1, 1], padding='SAME'),
                                 conv4_2_b)
        relu4_2 = tf.nn.relu(conv4_2)
        tf.summary.histogram("weights", conv4_2_W)
        tf.summary.histogram("bias", conv4_2_b)
        tf.summary.histogram("relu", relu4_2)

    with tf.name_scope("conv4_3"):
        conv4_3_W = tf.constant(trainedWeights[20], name="W")
        conv4_3_b = tf.constant(trainedWeights[21], name="b")
        conv4_3 = tf.nn.bias_add(tf.nn.conv2d(relu4_2, conv4_3_W, strides=[1, 1, 1, 1], padding='SAME'),
                                 conv4_3_b)
        relu4_3 = tf.nn.relu(conv4_3)
        tf.summary.histogram("weights", conv4_3_W)
        tf.summary.histogram("bias", conv4_3_b)
        tf.summary.histogram("relu", relu4_3)

    with tf.name_scope("conv4_4"):
        conv4_4_W = tf.constant(trainedWeights[22], name="W")
        conv4_4_b = tf.constant(trainedWeights[23], name="b")
        conv4_4 = tf.nn.bias_add(tf.nn.conv2d(relu4_3, conv4_4_W, strides=[1, 1, 1, 1], padding='SAME'),
                                 conv4_4_b)
        relu4_4 = tf.nn.relu(conv4_4)
        tf.summary.histogram("weights", conv4_4_W)
        tf.summary.histogram("bias", conv4_4_b)
        tf.summary.histogram("relu", relu4_4)

    with tf.name_scope("pool4"):
        pool4 = tf.nn.pool(relu4_4, [2, 2], "AVG", "VALID", strides=[2, 2])

    with tf.name_scope("conv5_1"):
        conv5_1_W = tf.constant(trainedWeights[24], name="W")
        conv5_1_b = tf.constant(trainedWeights[25], name="b")
        conv5_1 = tf.nn.bias_add(tf.nn.conv2d(pool4, conv5_1_W, strides=[1, 1, 1, 1], padding='SAME'),
                                 conv5_1_b)
        relu5_1 = tf.nn.relu(conv5_1)
        tf.summary.histogram("weights", conv5_1_W)
        tf.summary.histogram("bias", conv5_1_b)
        tf.summary.histogram("relu", relu5_1)

    return {"relu1_1": relu1_1,
            "relu1_2": relu1_2,
            "relu2_1": relu2_1,
            "relu2_2": relu2_2,
            "relu3_1": relu3_1,
            "relu3_2": relu3_2,
            "relu3_3": relu3_3,
            "relu3_4": relu3_4,
            "relu4_1": relu4_1,
            "relu4_2": relu4_2,
            "relu4_3": relu4_3,
            "relu4_4": relu4_4,
            "relu5_1": relu5_1,
            }







# Get activations for content and style images
a = tf.constant(getImage("style.jpg"))
tf.summary.histogram("style", a)
A = getModel(a, trainedWeights)

p = tf.constant(getImage("content.jpg"))
tf.summary.histogram("content", p)
P = getModel(p, trainedWeights)

# Build NN to optimize generated image
x = getRandomTensorForVGG()
tf.summary.histogram("generated", x)
F = getModel(x, trainedWeights)

preview = tf.concat([x, p, a], axis=2)[..., ::-1]
tf.summary.image("preview", preview, max_outputs=1)

with tf.name_scope("loss"):
    alpha = 10e-3
    beta = 1
    contentLoss = tf.nn.l2_loss(F["relu4_2"] - P["relu4_2"])
    styleLoss = (StyleLossForLayer(F["relu1_1"], A["relu1_1"]) / 5 +
                 StyleLossForLayer(F["relu2_1"], A["relu2_1"]) / 5 +
                 StyleLossForLayer(F["relu3_1"], A["relu3_1"]) / 5 +
                 StyleLossForLayer(F["relu4_1"], A["relu4_1"]) / 5 +
                 StyleLossForLayer(F["relu5_1"], A["relu5_1"]) / 5)
    loss = alpha * contentLoss + beta * styleLoss
    tf.summary.scalar("loss", loss)

optimizer = tf.train.AdamOptimizer(5).minimize(loss, var_list=[x])

config = tf.ConfigProto(device_count={'GPU': 1}, log_device_placement=True)
with tf.Session(config=config) as sess:
    writer = tf.summary.FileWriter("logs", sess.graph)
    summaryOp = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())

    for epoch in range(5000):
        _, summary = sess.run([optimizer, summaryOp])
        writer.add_summary(summary, epoch)
