import tensorflow as tf

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

