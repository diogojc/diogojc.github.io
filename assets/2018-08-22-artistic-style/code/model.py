import tensorflow as tf


def getActivations(X, weights, debug=False):
    # Block 1
    with tf.name_scope("conv1_1"):
        conv1_1_W = tf.constant(weights[0], name="W")
        conv1_1_b = tf.constant(weights[1], name="b")
        conv1_1 = tf.nn.conv2d(X,
                               conv1_1_W,
                               strides=[1, 1, 1, 1],
                               padding='SAME') + conv1_1_b
        relu1_1 = tf.nn.relu(conv1_1)
        if debug:
            tf.summary.histogram("weights", conv1_1_W)
            tf.summary.histogram("bias", conv1_1_b)
            tf.summary.histogram("relu", relu1_1)

    with tf.name_scope("conv1_2"):
        conv1_2_W = tf.constant(weights[2], name="W")
        conv1_2_b = tf.constant(weights[3], name="b")
        conv1_2 = tf.nn.conv2d(relu1_1,
                               conv1_2_W,
                               strides=[1, 1, 1, 1],
                               padding='SAME') + conv1_2_b
        relu1_2 = tf.nn.relu(conv1_2)
        if debug:
            tf.summary.histogram("weights", conv1_2_W)
            tf.summary.histogram("bias", conv1_2_b)
            tf.summary.histogram("relu", relu1_2)

    with tf.name_scope("pool1"):
        pool1 = tf.nn.pool(relu1_2, [2, 2], "AVG", "VALID", strides=[2, 2])

    # Block 2
    with tf.name_scope("conv2_1"):
        conv2_1_W = tf.constant(weights[4], name="W")
        conv2_1_b = tf.constant(weights[5], name="b")
        conv2_1 = tf.nn.conv2d(pool1,
                               conv2_1_W,
                               strides=[1, 1, 1, 1],
                               padding='SAME') + conv2_1_b
        relu2_1 = tf.nn.relu(conv2_1)
        if debug:
            tf.summary.histogram("weights", conv2_1_W)
            tf.summary.histogram("bias", conv2_1_b)
            tf.summary.histogram("relu", relu2_1)

    with tf.name_scope("conv2_2"):
        conv2_2_W = tf.constant(weights[6], name="W")
        conv2_2_b = tf.constant(weights[7], name="b")
        conv2_2 = tf.nn.conv2d(relu2_1,
                               conv2_2_W,
                               strides=[1, 1, 1, 1],
                               padding='SAME') + conv2_2_b
        relu2_2 = tf.nn.relu(conv2_2)
        if debug:
            tf.summary.histogram("weights", conv2_2_W)
            tf.summary.histogram("bias", conv2_2_b)
            tf.summary.histogram("relu", relu2_2)

    with tf.name_scope("pool2"):
        pool2 = tf.nn.pool(relu2_2, [2, 2], "AVG", "VALID", strides=[2, 2])

    # Block 3
    with tf.name_scope("conv3_1"):
        conv3_1_W = tf.constant(weights[8], name="W")
        conv3_1_b = tf.constant(weights[9], name="b")
        conv3_1 = tf.nn.conv2d(pool2,
                               conv3_1_W,
                               strides=[1, 1, 1, 1],
                               padding='SAME') + conv3_1_b
        relu3_1 = tf.nn.relu(conv3_1)
        if debug:
            tf.summary.histogram("weights", conv3_1_W)
            tf.summary.histogram("bias", conv3_1_b)
            tf.summary.histogram("relu", relu3_1)

    with tf.name_scope("conv3_2"):
        conv3_2_W = tf.constant(weights[10], name="W")
        conv3_2_b = tf.constant(weights[11], name="b")
        conv3_2 = tf.nn.conv2d(relu3_1,
                               conv3_2_W,
                               strides=[1, 1, 1, 1],
                               padding='SAME') + conv3_2_b
        relu3_2 = tf.nn.relu(conv3_2)
        if debug:
            tf.summary.histogram("weights", conv3_2_W)
            tf.summary.histogram("bias", conv3_2_b)
            tf.summary.histogram("relu", relu3_2)

    with tf.name_scope("conv3_3"):
        conv3_3_W = tf.constant(weights[12], name="W")
        conv3_3_b = tf.constant(weights[13], name="b")
        conv3_3 = tf.nn.conv2d(relu3_2,
                               conv3_3_W,
                               strides=[1, 1, 1, 1],
                               padding='SAME') + conv3_3_b
        relu3_3 = tf.nn.relu(conv3_3)
        if debug:
            tf.summary.histogram("weights", conv3_3_W)
            tf.summary.histogram("bias", conv3_3_b)
            tf.summary.histogram("relu", relu3_3)

    with tf.name_scope("conv3_4"):
        conv3_4_W = tf.constant(weights[14], name="W")
        conv3_4_b = tf.constant(weights[15], name="b")
        conv3_4 = tf.nn.conv2d(relu3_3,
                               conv3_4_W,
                               strides=[1, 1, 1, 1],
                               padding='SAME') + conv3_4_b
        relu3_4 = tf.nn.relu(conv3_4)
        if debug:
            tf.summary.histogram("weights", conv3_4_W)
            tf.summary.histogram("bias", conv3_4_b)
            tf.summary.histogram("relu", relu3_4)

    with tf.name_scope("pool3"):
        pool3 = tf.nn.pool(relu3_4, [2, 2], "AVG", "VALID", strides=[2, 2])

    # Block 4
    with tf.name_scope("conv4_1"):
        conv4_1_W = tf.constant(weights[16], name="W")
        conv4_1_b = tf.constant(weights[17], name="b")
        conv4_1 = tf.nn.conv2d(pool3,
                               conv4_1_W,
                               strides=[1, 1, 1, 1],
                               padding='SAME') + conv4_1_b
        relu4_1 = tf.nn.relu(conv4_1)
        if debug:
            tf.summary.histogram("weights", conv4_1_W)
            tf.summary.histogram("bias", conv4_1_b)
            tf.summary.histogram("relu", relu4_1)

    with tf.name_scope("conv4_2"):
        conv4_2_W = tf.constant(weights[18], name="W")
        conv4_2_b = tf.constant(weights[19], name="b")
        conv4_2 = tf.nn.conv2d(relu4_1,
                               conv4_2_W,
                               strides=[1, 1, 1, 1],
                               padding='SAME') + conv4_2_b
        relu4_2 = tf.nn.relu(conv4_2)
        if debug:
            tf.summary.histogram("weights", conv4_2_W)
            tf.summary.histogram("bias", conv4_2_b)
            tf.summary.histogram("relu", relu4_2)

    with tf.name_scope("conv4_3"):
        conv4_3_W = tf.constant(weights[20], name="W")
        conv4_3_b = tf.constant(weights[21], name="b")
        conv4_3 = tf.nn.conv2d(relu4_2,
                               conv4_3_W,
                               strides=[1, 1, 1, 1],
                               padding='SAME') + conv4_3_b
        relu4_3 = tf.nn.relu(conv4_3)
        if debug:
            tf.summary.histogram("weights", conv4_3_W)
            tf.summary.histogram("bias", conv4_3_b)
            tf.summary.histogram("relu", relu4_3)

    with tf.name_scope("conv4_4"):
        conv4_4_W = tf.constant(weights[22], name="W")
        conv4_4_b = tf.constant(weights[23], name="b")
        conv4_4 = tf.nn.conv2d(relu4_3,
                               conv4_4_W,
                               strides=[1, 1, 1, 1],
                               padding='SAME') + conv4_4_b
        relu4_4 = tf.nn.relu(conv4_4)
        if debug:
            tf.summary.histogram("weights", conv4_4_W)
            tf.summary.histogram("bias", conv4_4_b)
            tf.summary.histogram("relu", relu4_4)

    with tf.name_scope("pool4"):
        pool4 = tf.nn.pool(relu4_4, [2, 2], "AVG", "VALID", strides=[2, 2])

    # Block 5
    with tf.name_scope("conv5_1"):
        conv5_1_W = tf.constant(weights[24], name="W")
        conv5_1_b = tf.constant(weights[25], name="b")
        conv5_1 = tf.nn.conv2d(pool4,
                               conv5_1_W,
                               strides=[1, 1, 1, 1],
                               padding='SAME') + conv5_1_b
        relu5_1 = tf.nn.relu(conv5_1)
        if debug:
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
            "relu5_1": relu5_1}
