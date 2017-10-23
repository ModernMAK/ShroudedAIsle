import tensorflow as tf


def weight_variable(weight_shape, name="weights"):
    # Avoids symmetry when training, and prevents 0 gradients
    initial = tf.truncated_normal(weight_shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(bias_shape, name="biases"):
    # Avoids dead neurons
    initial = tf.constant(0.1, shape=bias_shape)
    return tf.Variable(initial, name=name)


def conv2d(x, W, name="conv"):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME", name=name)


def max_pool_2x2(x, name="pool"):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name=name)

class ImageGraph:
    def logits_from_inn(self, inn, *, params={}):
        return self.logits(inn.images, inn.probability, inn.shape, inn.channels, inn.categories, params=params)

    def logits(self, images, probability, image_shape, channels, categories, *, params={}):
        flat_shape = int(image_shape[0] * image_shape[1] / 16)
        conv1_nuerons = params.get("conv1", 32)
        conv2_nuerons = params.get("conv2", 64)
        fc1_nuerons = params.get("fc1", 1024)

        images = tf.reshape(images, [-1, image_shape[0], image_shape[1], channels])
        with tf.name_scope("conv1"):
            w_conv1 = weight_variable([5, 5, channels, conv1_nuerons])
            b_conv1 = bias_variable([conv1_nuerons])
            conv1 = tf.nn.relu(conv2d(images, w_conv1) + b_conv1)

        with tf.name_scope("pool1"):
            pool1 = max_pool_2x2(conv1)

        with tf.name_scope("conv2"):
            w_conv2 = weight_variable([5, 5, conv1_nuerons, conv2_nuerons])
            b_conv2 = bias_variable([conv2_nuerons])
            conv2 = tf.nn.relu(conv2d(pool1, w_conv2) + b_conv2)

        with tf.name_scope("pool2"):
            pool2 = max_pool_2x2(conv2)

        with tf.name_scope("fc1"):
            size = int(flat_shape * conv2_nuerons)

            w_fc1 = weight_variable([size, fc1_nuerons])
            b_fc1 = bias_variable([fc1_nuerons])

            pool2_flat = tf.reshape(pool2, [-1, size])
            fc1 = tf.nn.relu(tf.matmul(pool2_flat, w_fc1) + b_fc1)

        with tf.name_scope("dropouts"):
            dropouts = tf.nn.dropout(fc1, probability)

        with tf.name_scope("logits"):
            w_logits = weight_variable([fc1_nuerons, categories])
            b_logits = bias_variable([categories])

            logits = tf.matmul(dropouts, w_logits) + b_logits
            return logits

    def loss_from_inn(self, inn, logits):
        return self.loss(inn.labels, logits)

    def loss(self, labels, logits):
        with tf.name_scope("loss"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=labels,
                logits=logits,
                name="cross_entropy")
            cross_entropy_mean = tf.reduce_mean(cross_entropy, name="cross_entropy_mean")
            return cross_entropy_mean

    def training(self, loss, *, params):
        learning_step = params.get("learning_rate", 0.001)
        return tf.train.AdamOptimizer(learning_step).minimize(loss)

    def evaluation_from_inn(self, inn, logits):
        return self.evaluation(inn.labels, logits)

    def evaluation(self, labels, logits):
        with tf.name_scope("evaluation"):
            max_logits = tf.argmax(logits, 1, name="logit_max")
            max_labels = tf.argmax(labels, 1, name="label_max")
            correct_prediction = tf.equal(max_logits, max_labels)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            return accuracy