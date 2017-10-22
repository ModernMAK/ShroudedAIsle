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


class ImageNeuralNetwork:
    def __init__(self, image_shape, channels, categories):
        self.shape = image_shape
        self.channels = channels
        self.categories = categories
        self.flat_shape = image_shape[0] * image_shape[1] * channels
        with tf.name_scope("inputs"):
            self.images = tf.placeholder(tf.float32, shape=[None, self.flat_shape], name="images")
            self.labels = tf.placeholder(tf.float32, shape=[None, categories], name="labels")
            self.logits = self.build_logits()
            self.loss = self.build_loss()
            self.training = self.build_training()
            self.evaluation = self.build_evaluation()

    def feed_dict(self, images, labels):
        dict = {
            self.images : images,
            self.labels : labels
        }
        return dict

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

    def build_logits(self):
        return tf.constant(tf.float32, 0)

    def build_loss(self):
        return tf.constant(tf.float32, 1)

    def build_training(self):
        return tf.constant(tf.float32, 0)

    def build_evaluation(self):
        return tf.constant(tf.float32, 0)


class GenderBrain(ImageNeuralNetwork):
    def __init__(self, image_shape, channels, categories, probability):
        self.probability = probability
        super().__init__(image_shape, channels, categories)

    def build_logits(self):
        conv1_nuerons = 32
        conv2_nuerons = 64
        fcl_nuerons = 64

        images = tf.reshape(self.images, [-1, self.shape[0], self.shape[1], self.channels])

        with tf.name_scope("conv1"):
            w_conv1 = self.weight_variable([5, 5, self.channels, conv1_nuerons])
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
            size = int(self.flat_shape / 3 / 16 * conv2_nuerons)

            w_fc1 = self.weight_variable([size, fcl_nuerons])
            b_fc1 = self.bias_variable([fcl_nuerons])

            pool2_flat = tf.reshape(pool2, [-1, size])
            fc1 = tf.nn.relu(tf.matmul(pool2_flat, w_fc1) + b_fc1)

        with tf.name_scope("dropouts"):
            dropouts = tf.nn.dropout(fc1, self.probability)

        with tf.name_scope("logits"):
            w_logits = self.weight_variable([fcl_nuerons, self.categories])
            b_logits = self.bias_variable([self.categories])

            logits = tf.matmul(dropouts, w_logits) + b_logits
            return logits

    def build_loss(self):
        with tf.name_scope("loss"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=self.labels,
                logits=self.logits,
                name="cross_entropy")
            cross_entropy_mean = tf.reduce_mean(cross_entropy, name="cross_entropy_mean")
            return cross_entropy_mean

    def build_evaluation(self):
        with tf.name_scope("evaluation"):
            max_logits = tf.argmax(self.logits, 1, name="logit_max")
            max_labels = tf.argmax(self.labels, 1, name="label_max")
            correct_prediction = tf.equal(max_logits, max_labels)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            return accuracy
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

    def build_training(self):
        return tf.train.AdamOptimizer(0.01).minimize(self.loss)
