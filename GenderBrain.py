from GenderAI import GenderDataset
from GenderAI import DisposableGenderDataset
import tensorflow as tf
import numpy as np
import math


def load_dataset_from_directory(directory):
    dataset = GenderDataset()
    dataset.load_from_images(directory, [128, 128], 3)
    return dataset


def load_dataset_from_file(file, mode):
    dataset = GenderDataset()
    dataset.load_from_file(file, mode=mode)
    return dataset


class ImageGraph:
    def weight_variable(self, weight_shape, name="weights"):
        # Avoids symmetry when training, and prevents 0 gradients
        initial = tf.truncated_normal(weight_shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    def bias_variable(self, bias_shape, name="biases"):
        # Avoids dead neurons
        initial = tf.constant(0.1, shape=bias_shape)
        return tf.Variable(initial, name=name)

    def conv2d(self, x, W, name="conv"):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME", name=name)

    def max_pool_2x2(self, x, name="pool"):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name=name)

    def logits_from_INN(self, inn, *, params={}):
        return self.logits(inn.images, inn.probability, inn.shape, inn.channels, inn.categories, params=params)

    def logits(self, images, probability, image_shape, channels, categories, *, params={}):
        flat_shape = int(image_shape[0] * image_shape[1] / 16)
        conv1_nuerons = params.get("conv1", 32)
        conv2_nuerons = params.get("conv2", 64)
        fc1_nuerons = params.get("fc1", 1024)

        images = tf.reshape(images, [-1, image_shape[0], image_shape[1], channels])
        with tf.name_scope("conv1"):
            w_conv1 = self.weight_variable([5, 5, channels, conv1_nuerons])
            b_conv1 = self.bias_variable([conv1_nuerons])
            conv1 = tf.nn.relu(self.conv2d(images, w_conv1) + b_conv1)

        with tf.name_scope("pool1"):
            pool1 = self.max_pool_2x2(conv1)

        with tf.name_scope("conv2"):
            w_conv2 = self.weight_variable([5, 5, conv1_nuerons, conv2_nuerons])
            b_conv2 = self.bias_variable([conv2_nuerons])
            conv2 = tf.nn.relu(self.conv2d(pool1, w_conv2) + b_conv2)

        with tf.name_scope("pool2"):
            pool2 = self.max_pool_2x2(conv2)

        with tf.name_scope("fc1"):
            size = int(flat_shape * conv2_nuerons)

            w_fc1 = self.weight_variable([size, fc1_nuerons])
            b_fc1 = self.bias_variable([fc1_nuerons])

            pool2_flat = tf.reshape(pool2, [-1, size])
            fc1 = tf.nn.relu(tf.matmul(pool2_flat, w_fc1) + b_fc1)

        with tf.name_scope("dropouts"):
            dropouts = tf.nn.dropout(fc1, probability)

        with tf.name_scope("logits"):
            w_logits = self.weight_variable([fc1_nuerons, categories])
            b_logits = self.bias_variable([categories])

            logits = tf.matmul(dropouts, w_logits) + b_logits
            return logits

    def loss_from_INN(self, inn, logits):
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

    def evaluation_from_INN(self, inn, logits):
        return self.evaluation(inn.labels, logits)

    def evaluation(self, labels, logits):
        with tf.name_scope("evaluation"):
            max_logits = tf.argmax(logits, 1, name="logit_max")
            max_labels = tf.argmax(labels, 1, name="label_max")
            correct_prediction = tf.equal(max_logits, max_labels)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            return accuracy


class ImageNeuralNetwork:
    def __init__(self, image_shape, channels, categories):
        self.shape = image_shape
        self.channels = channels
        self.categories = categories
        self.flat_shape = image_shape[0] * image_shape[1] * channels
        with tf.name_scope("inputs"):
            self.probability = tf.placeholder(tf.float32)
            self.images = tf.placeholder(tf.float32, shape=[None, self.flat_shape], name="images")
            self.labels = tf.placeholder(tf.float32, shape=[None, categories], name="labels")
        self.logits = None
        self.loss = None
        self.training = None
        self.evaluation = None

    def feed_dict(self, images, labels, probability):
        dict = {
            self.images: images,
            self.labels: labels,
            self.probability: probability
        }
        return dict

    def build_graph(self, graph, *, params={}):
        # Why not just use self and infer logits/loss
        # Honestly... I dont know
        self.logits = graph.logits_from_INN(self, params=params)
        self.loss = graph.loss_from_INN(self, self.logits)
        self.training = graph.training(self.loss, params=params)
        self.evaluation = graph.evaluation_from_INN(self, self.logits)





        #
        #
        #
        # # Params is a dict with
        # # params['shape'] = [int,int] (REQ)
        # # params['channels] = int (REQ)
        # # params['genders'] = int
        # def inputs(self, params):
        #     shape = params['shape']
        #     channels = params['channels']
        #     genders = params.get('genders', 2)
        #     with tf.name_scope('inputs'):
        #         image_pixel_count = shape[0] * shape[1] * channels
        #         images_placeholder = tf.placeholder(tf.float32, shape=[None, image_pixel_count], name="images")
        #         labels_placeholder = tf.placeholder(tf.float32, shape=[None, genders], name="labels")
        #         return images_placeholder, labels_placeholder
        #
        # def weight_variable(self, weight_shape, name="weights"):
        #     # Avoids symmetry when training, and prevents 0 gradients
        #     initial = tf.truncated_normal(weight_shape, stddev=0.1)
        #     return tf.Variable(initial, name=name)
        #
        # def bias_variable(self, bias_shape, name="biases"):
        #     # Avoids dead neurons
        #     initial = tf.constant(0.1, shape=bias_shape)
        #     return tf.Variable(initial, name=name)
        #
        # def conv2d(self, x, W, name="conv"):
        #     return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME", name=name)
        #
        # def max_pool_2x2(self, x, name="pool"):
        #     return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name=name)
        #
        # # Params is a dict with
        # # params['conv1'] = int
        # # params['conv2'] = int
        # # params['fcl'] = int
        # # params['shape'] = [int,int] (REQ)
        # # params['channels] = int (REQ)
        # # params['genders'] = int
        # # params['keep'] = float
        #
        # def inference(self, images, params={}):
        #     conv1_neurons = params.get('conv1', 32)
        #     conv2_neurons = params.get('conv2', 64)
        #     fully_connected_neurons = params.get('fcl', 1024)
        #     shape = params['shape']
        #     channels = params['channels']
        #     genders = params.get('genders', 2)
        #     keep_prob = params.get("keep", 1.0)
        #
        #     with tf.name_scope('reshape'):
        #         x = tf.reshape(images, [-1, shape[0], shape[1], channels])
        #
        #     with tf.name_scope('conv1'):
        #         weights = self.weight_variable([5, 5, channels, conv1_neurons])
        #         biases = self.bias_variable([conv1_neurons])
        #         conv1 = tf.nn.relu(self.conv2d(x, weights) + biases)
        #
        #     with tf.name_scope('pool1'):
        #         pool1 = self.max_pool_2x2(conv1)
        #
        #     with tf.name_scope('conv2'):
        #         weights = self.weight_variable([5, 5, conv1_neurons, conv2_neurons])
        #         biases = self.bias_variable([conv2_neurons])
        #         conv2 = tf.nn.relu(self.conv2d(pool1, weights) + biases)
        #
        #     with tf.name_scope('pool2'):
        #         pool2 = self.max_pool_2x2(conv2)
        #
        #     with tf.name_scope('dense'):
        #         flat_shape = int(shape[0] * shape[1] * channels / 16) * conv2_neurons
        #         weights = self.weight_variable([flat_shape, fully_connected_neurons])
        #         biases = self.bias_variable([fully_connected_neurons])
        #
        #         pool_flat = tf.reshape(pool2, [-1, flat_shape], name="pool_flat")
        #         dense = tf.nn.relu(tf.matmul(pool_flat, weights) + biases, name="dense")
        #
        #     with tf.name_scope('drop'):
        #         dropouts = tf.nn.dropout(dense, keep_prob, name="dropouts")
        #
        #     with tf.name_scope("logits"):
        #         weights = self.weight_variable([fully_connected_neurons, genders])
        #         biases = self.bias_variable([genders])
        #
        #         logits = tf.matmul(dropouts, weights) + biases
        #
        #     return logits
        #
        # def loss(self, logits, labels):
        #     with tf.name_scope('loss'):
        #         print(tf.shape(logits))
        #         print(tf.shape(labels))
        #         cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #             labels=labels,
        #             logits=logits,
        #             name="cross_entropy"
        #         )
        #         return tf.reduce_mean(cross_entropy, name="cross_entropy_mean")
        #
        # def training(self, loss, learning_rate):
        #     with tf.name_scope("training"):
        #         tf.summary.scalar('loss', loss)
        #         optimizer = tf.train.AdamOptimizer(learning_rate)
        #         global_step = tf.Variable(0, name='global_step', trainable=False)
        #         train_op = optimizer.minimize(loss, global_step=global_step)
        #         return train_op
        #
        # def evaluation(self, logits, labels):
        #     with tf.name_scope("evaluation"):
        #         labels = tf.reshape(labels, [None, self.genders])
        #         max_logits = tf.argmax(logits, 1, name="logits_max")
        #         max_labels = tf.argmax(labels, 1, name="labels_max")
        #
        #         correct_prediction = tf.equal(max_logits, max_labels)
        #         accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
