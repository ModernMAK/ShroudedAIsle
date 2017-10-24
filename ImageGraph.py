import tensorflow as tf


def build_convolutional_neural_network(inn, scope="cnn", params={}):
    input = inn.images
    labels = inn.labels
    probability = inn.probability
    image_shape = inn.shape
    channels = inn.channels
    categories = inn.categories
    with inn.graph.as_default() as g:
        with g.name_scope(scope) as scope:
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

            def inference():
                with tf.name_scope("inference"):
                    flat_shape = int(image_shape[0] * image_shape[1] / 16)
                    conv1_nuerons = params.get("conv1", 32)
                    conv2_nuerons = params.get("conv2", 64)
                    fc1_nuerons = params.get("fc1", 1024)

                    images = tf.reshape(input, [-1, image_shape[0], image_shape[1], channels])
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
                        fc1 = tf.nn.relu(tf.matmul(pool2_flat, w_fc1) + b_fc1, name="dense")

                    with tf.name_scope("dropouts"):
                        dropouts = tf.nn.dropout(fc1, probability, name="dropout")

                    with tf.name_scope("inference"):
                        w_logits = weight_variable([fc1_nuerons, categories])
                        b_logits = bias_variable([categories])

                        logits = tf.identity(tf.matmul(dropouts, w_logits) + b_logits, "inference")
                        return logits

            def loss(logits):
                with tf.name_scope("loss"):
                    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                        labels=labels,
                        logits=logits,
                        name="cross_entropy")
                    cross_entropy_mean = tf.reduce_mean(cross_entropy, name="cross_entropy_mean")
                    return cross_entropy_mean

            def training(loss):
                with tf.name_scope("training"):
                    learning_step = params.get("learning_rate", 0.001)
                    return tf.train.AdamOptimizer(learning_step, name="adam_training").minimize(loss)

            def evaluation(logits):
                with tf.name_scope("evaluation"):
                    max_logits = tf.argmax(logits, 1, name="logit_max")
                    max_labels = tf.argmax(labels, 1, name="label_max")
                    correct_prediction = tf.equal(max_logits, max_labels, name="logit_label_match")
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

                    return accuracy

            def predict(logits):
                with tf.name_scope("prediction"):
                    return tf.nn.softmax(logits, name="predictions")

            inn.t_inference = inference()
            inn.t_loss = loss(inn.t_inference)
            inn.t_evaluation = evaluation(inn.t_inference)
            inn.t_training = training(inn.t_loss)
            inn.t_predict = predict(inn.t_inference)
