# # import tensorflow as tf
# # from tensorflow.contrib.data import Dataset
# # import numpy as np
# # from PIL import Image
# # from os import walk
# # from os.path import join, basename
# #
# #
# # def main():
# #     print("main")
# #
# #     def load_dataset(dataset_dir):
# #         input_params = {
# #             "filenames": [],
# #             "labels": [],
# #         }
# #         for (dirpath, dirnames, filenames) in walk(dataset_dir):
# #             for file in filenames:
# #                 input_params["filenames"].append(join(dirpath, file))
# #                 input_params["labels"].append(int(basename(dirpath)))
# #
# #         return create_dataset(input_params)
# #
# #     def parse_image(filename, label, channels=3, size=[128,108]):
# #         image_string = tf.read_file(filename)
# #         image_decoded = tf.image.decode_image(image_string, channels=channels)
# #         image_decoded_float = tf.cast(image_decoded, tf.float32)
# #         image_decoded_normalized = tf.divide(image_decoded_float, tf.constant(255.0, dtype=tf.float32))
# #         # image_decoded_normalized_flat = tf.reshape(image_decoded_normalized, [-1])
# #         # image_resized = tf.image.resize_image_with_crop_or_pad(image_decoded, size, size)
# #         print(image_decoded_normalized)
# #         return image_decoded_normalized, label
# #
# #     def create_dataset(params):
# #         filenames_node = tf.constant(params["filenames"], dtype=tf.string)
# #         labels_node = tf.constant(params["labels"], dtype=tf.int32)
# #         # labels_onehot_node = tf.one_hot(labels_node, 2)
# #         # labels_float_onehot = tf.cast(labels_onehot_node, dtype=tf.float32)
# #         dataset = tf.contrib.data.Dataset.from_tensor_slices((filenames_node, labels_node))
# #         dataset = dataset.map(parse_image)
# #         dataset = dataset.repeat()
# #         dataset = dataset.get_full_batch(8)
# #
# #         return dataset
# #
# #     dataset = load_dataset("Training/images")
# #
# #     #
# #     # in_size = 128 * 108 * 3
# #     # out_size = 2
# #     # #
# #     # # x = tf.reshape(x, [None, in_size])
# #     # x = tf.placeholder(tf.float32, [None, in_size])
# #     # w = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.001))
# #     # b = tf.Variable(tf.zeros([out_size]))
# #     # y = tf.nn.softmax(tf.matmul(x, w) + b)
# #     #
# #     # y_ = tf.placeholder(tf.float32, [None, out_size])
# #     # # y_ = tf.reshape(y_, [None, out_size])
# #     # cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# #     # train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)
# #     #
# #     # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# #     # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# #
# #     def model(batch_size, input_images, input_labels, mode):
# #         input_images = tf.reshape(input_images, [batch_size, 128, 108, 3])
# #
# #         print(input_images)
# #         print(tf.shape(input_images))
# #         conv1 = tf.layers.conv2d(
# #             inputs=input_images,
# #             filters=32,
# #             kernel_size=[5, 5],
# #             padding="same",
# #             activation=tf.nn.relu
# #         )
# #         print(conv1)
# #         print(tf.shape(conv1))
# #         pool1 = tf.layers.max_pooling2d(
# #             inputs=conv1,
# #             pool_size=[2, 2],
# #             strides=2
# #         )
# #         print(pool1)
# #         print(tf.shape(pool1))
# #         conv2 = tf.layers.conv2d(
# #             inputs=pool1,
# #             filters=64,
# #             kernel_size=[5, 5],
# #             padding="same",
# #             activation=tf.nn.relu
# #         )
# #         print(conv2)
# #         print(tf.shape(conv2))
# #         pool2 = tf.layers.max_pooling2d(
# #             inputs=conv2,
# #             pool_size=[2, 2],
# #             strides=2
# #         )
# #         print(pool2)
# #         print(tf.shape(pool2))
# #         pool2_flat = tf.reshape(pool2, [batch_size, -1])
# #         print(pool2_flat)
# #         print(tf.shape(pool2_flat))
# #         dense = tf.layers.dense(
# #             inputs=pool2_flat,
# #             units=1024,
# #             activation=tf.nn.relu
# #         )
# #         print(dense)
# #         print(tf.shape(dense))
# #         dropout = tf.layers.dropout(
# #             inputs=dense,
# #             rate=0.4,
# #             training=(mode == tf.estimator.ModeKeys.TRAIN)
# #         )
# #         print(dropout)
# #         print(tf.shape(dropout))
# #         logits = tf.layers.dense(inputs=dropout, units=2)
# #         print(logits)
# #         print(tf.shape(logits))
# #         predictions = {
# #             "classes": tf.argmax(input=logits, axis=1),
# #             "probabilities": tf.nn.softmax(logits, name="softmax")
# #         }
# #
# #         if mode == tf.estimator.ModeKeys.PREDICT:
# #             return tf.estimator.EstimatorSpec(
# #                 mode=mode,
# #                 predictions=predictions)
# #
# #         onehot_labels = tf.one_hot(indices=tf.cast(input_labels, tf.int32), depth=2)
# #         loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
# #
# #         if mode == tf.estimator.ModeKeys.TRAIN:
# #             optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
# #             train_op = optimizer.minimize(
# #                 loss=loss,
# #                 global_step=tf.train.get_global_step()
# #             )
# #             return tf.estimator.EstimatorSpec(
# #                 mode=mode,
# #                 loss=loss,
# #                 train_op=train_op
# #             )
# #
# #         eval_metric_ops = {
# #             "accuracy": tf.metrics.accuracy(
# #                 labels=labels, predictions=predictions["classes"]
# #             )
# #         }
# #         return tf.estimator.EstimatorSpec(
# #             mode=mode,
# #             loss=loss,
# #             eval_metric_ops=eval_metric_ops
# #         )
# #
# #     with tf.Session() as sess:
# #         tf.global_variables_initializer().run()
# #         iterator = dataset.make_initializable_iterator()
# #         images, labels = iterator.get_next()
# #         print("A")
# #         for _ in range(100):
# #             sess.run(iterator.initializer)
# #             while True:
# #                 try:
# #                     sess.run(model(8, images, labels, mode=tf.estimator.ModeKeys.TRAIN),feed_dict={})
# #                 except tf.errors.OutOfRangeError as e:
# #                     # print(e)
# #                     break
# #             sess.run(iterator.initializer)
# #             while True:
# #                 try:
# #                     sess.run(model(8, images, labels, mode=tf.estimator.ModeKeys.PREDICT),feed_dict={})
# #                 except tf.errors.OutOfRangeError as e:
# #                     # print(e)
# #                     break
# #         print("B")
# import tensorflow as tf
# import numpy as np
# from os import walk
# from os.path import basename, join
#
#
# def cnn_model_fn(features, labels, mode):
#     """Model function for CNN."""
#     # Input Layer
#     # Reshape X to 4-D tensor: [batch_size, width, height, channels]
#     # MNIST images are 28x28 pixels, and have one color channel
#     input_layer = tf.reshape(features["x"], [-1, 128, 108, 3])
#
#     # Convolutional Layer #1
#     # Computes 32 features using a 5x5 filter with ReLU activation.
#     # Padding is added to preserve width and height.
#     # Input Tensor Shape: [batch_size, 128, 108, 3]
#     # Output Tensor Shape: [batch_size, 128, 108, 32]
#     conv1 = tf.layers.conv2d(
#         inputs=input_layer,
#         filters=32,
#         kernel_size=[5, 5],
#         padding="same",
#         activation=tf.nn.relu)
#
#     # Pooling Layer #1
#     # First max pooling layer with a 2x2 filter and stride of 2
#     # Input Tensor Shape: [batch_size, 108, 128, 32]
#     # Output Tensor Shape: [batch_size, 54, 64, 32]
#     pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
#
#     # Convolutional Layer #2
#     # Computes 64 features using a 5x5 filter.
#     # Padding is added to preserve width and height.
#     # Input Tensor Shape: [batch_size, 64, 54, 32]
#     # Output Tensor Shape: [batch_size, 64, 54, 64]
#     conv2 = tf.layers.conv2d(
#         inputs=pool1,
#         filters=64,
#         kernel_size=[5, 5],
#         padding="same",
#         activation=tf.nn.relu)
#
#     # Pooling Layer #2
#     # Second max pooling layer with a 2x2 filter and stride of 2
#     # Input Tensor Shape: [batch_size, 54, 64, 64]
#     # Output Tensor Shape: [batch_size, 27, 32, 64]
#     pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
#
#     # Flatten tensor into a get_full_batch of vectors
#     # Input Tensor Shape: [batch_size, 27, 32, 64]
#     # Output Tensor Shape: [batch_size, 32 * 27 * 64]
#     pool2_flat = tf.reshape(pool2, [-1, 32 * 27 * 64])
#
#     # Dense Layer
#     # Densely connected layer with 1024 neurons
#     # Input Tensor Shape: [batch_size, 32 * 27 * 64]
#     # Output Tensor Shape: [batch_size, 1024]
#     dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
#
#     # Add dropout operation; 0.6 probability that element will be kept
#     dropout = tf.layers.dropout(
#         inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
#
#     # Logits layer
#     # Input Tensor Shape: [batch_size, 1024]
#     # Output Tensor Shape: [batch_size, 2]
#     logits = tf.layers.dense(inputs=dropout, units=2)
#
#     predictions = {
#         # Generate predictions (for PREDICT and EVAL mode)
#         "classes": tf.argmax(input=logits, axis=1),
#         # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
#         # `logging_hook`.
#         "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
#     }
#     if mode == tf.estimator.ModeKeys.PREDICT:
#         return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
#
#     # Calculate Loss (for both TRAIN and EVAL modes)
#     onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
#     loss = tf.losses.softmax_cross_entropy(
#         onehot_labels=onehot_labels, logits=logits)
#
#     # Configure the Training Op (for TRAIN mode)
#     if mode == tf.estimator.ModeKeys.TRAIN:
#         optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
#         train_op = optimizer.minimize(
#             loss=loss,
#             global_step=tf.train.get_global_step())
#         return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
#
#     # Add evaluation metrics (for EVAL mode)
#     eval_metric_ops = {
#         "accuracy": tf.metrics.accuracy(
#             labels=labels, predictions=predictions["classes"])}
#     return tf.estimator.EstimatorSpec(
#         mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
#
#
# def main():
#     def fetch_data(data_dir):
#         def load_flat_image(filename):
#             # print(filename)
#             from PIL import Image
#             img = Image.open(filename).convert("RGB")
#             # print(img)
#             arr = np.array(img)
#             # print(np.shape(arr))
#             arr = arr.astype(np.float32)
#             # print(np.shape(arr))
#             arr_flat = arr.flatten()
#             # print(np.shape(arr_flat))
#             arr_normalized = np.divide(arr_flat, 255)
#
#             return np.reshape(arr_normalized, [128, 108, 3])
#
#         images = []
#         labels = []
#         for (dirpath, dirnames, filenames) in walk(data_dir):
#             for file in filenames:
#                 label = int(basename(dirpath))
#                 full_file_path = join(dirpath, file)
#                 flat_image = load_flat_image(full_file_path)
#                 images.append(flat_image)
#                 labels.append(label)
#
#         print(images)
#         images = np.array(images)
#         print(images)
#
#         return {
#             "images": images,
#             "labels": labels
#         }
#
#     # Load training and eval data
#     training_dict = fetch_data("Training/images")
#     train_data = training_dict["images"]  # Returns np.array
#     train_labels = np.asarray(training_dict["labels"], dtype=np.int32)
#     eval_data = training_dict["images"]  # Returns np.array
#     eval_labels = np.asarray(training_dict["labels"], dtype=np.int32)
#
#     print(np.shape(train_data))
#
#     # Create the Estimator
#     mnist_classifier = tf.estimator.Estimator(
#         model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")
#
#     # Set up logging for predictions
#     # Log the values in the "Softmax" tensor with label "probabilities"
#     tensors_to_log = {"probabilities": "softmax_tensor"}
#     logging_hook = tf.train.LoggingTensorHook(
#         tensors=tensors_to_log, every_n_iter=50)
#
#     # Train the model
#     train_input_fn = tf.estimator.inputs.numpy_input_fn(
#         x={"x": train_data},
#         y=train_labels,
#         batch_size=1,
#         num_epochs=None,
#         shuffle=True)
#     mnist_classifier.train(
#         input_fn=train_input_fn,
#         steps=100,
#         hooks=[])
#
#     # Evaluate the model and print results
#     eval_input_fn = tf.estimator.inputs.numpy_input_fn(
#         x={"x": eval_data},
#         y=eval_labels,
#         num_epochs=1,
#         shuffle=False)
#     eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
#     print(eval_results)

import tensorflow as tf
import numpy as np
import os
from os import path
from PIL import Image


class GenderDataset:
    # Class Stuff
    def __init__(self):
        self.image_data = None
        self.gender_data = None
        self.batch_size = 0

    def __iter__(self):
        return [self.image_data, self.gender_data]

    def __getitem__(self, key):
        return [self.image_data, self.gender_data][key]

    # Functions

    def load_from_images(self, directory, shape, channels):
        images = []
        genders = []
        batch_size = 0

        for (dirpath, dirnames, filenames) in os.walk(directory):
            for file in filenames:
                filepath = path.join(dirpath, file)
                print(filepath)
                image, gender = self.get_image_and_label_from_image_and_reshape(filepath, shape, channels)
                image = image.flatten()
                gender = gender.flatten()
                images.append(image)
                genders.append(gender)
                batch_size += 1

        self.image_data = np.array(images)
        self.gender_data = np.array(genders)
        self.batch_size = batch_size

    def load_from_file(self, file):
        raise Exception("Unimplimented")

    def shuffle(self):
        p = np.random.permutation(len(self.image_data))
        result = GenderDataset()
        result.image_data = np.array(self.image_data[p])
        result.gender_data = np.array(self.gender_data[p])
        result.batch_size = self.batch_size
        return result

    def repeat(self, epochs=1):
        result = GenderDataset()
        img = []
        gen = []
        for i in range(epochs):
            img.extend(self.image_data)
            gen.extend(self.gender_data)
        result.image_data = np.array(img)
        result.gender_data = np.array(gen)
        result.batch_size = self.batch_size * epochs
        return result

    def get_full_batch(self):
        return self.get_batch(self.batch_size)

    def get_batch_from_epoch(self, batch_size, epoch=0):
        return self.get_batch(batch_size, batch_size * epoch)

    def get_batch(self, batch_size, offset=0):
        if batch_size + offset > self.batch_size:
            raise ValueError("No more batches can be made of batch_size")
        result = GenderDataset()

        img = []
        gen = []
        for count in range(batch_size):
            index = count + offset
            img.append(self.image_data[index])
            gen.append(self.gender_data[index])
        result.image_data = np.array(img)
        result.gender_data = np.array(gen)
        result.batch_size = batch_size

        return result

    # UTILITY FUNCS

    def get_image_and_label_from_image(self, file_path):
        gender = self.get_gender_from_image(file_path)
        image = self.get_image_from_file(file_path)
        gender_one_hot = self.get_one_hot(gender, 2)
        return image, gender_one_hot

    def get_image_and_label_from_image_and_reshape(self, file_path, shape, channels):
        image, gender = self.get_image_and_label_from_image(file_path)
        image = self.reshape_image(image, shape, channels)
        return image, gender

    def get_gender_from_image(self, file_path):
        folder_path = path.dirname(file_path)
        folder = path.basename(folder_path)

        male = ["Male", "male", "Men", "men", "M", "m"]
        female = ["Female", "female", "Women", "Women", "W", "w"]

        if folder in male:
            return 0
        if folder in female:
            return 1
        raise Exception("(%s) folder name invalid! From (%s)" % (folder, folder_path))

    # Returns a numpy array
    def get_image_from_file(self, file):
        pil_img = Image.open(file).convert("RGB")
        numpy_arr = np.array(pil_img)
        numpy_norm_arr = np.divide(numpy_arr, 255)
        return numpy_norm_arr

    def reshape_image(self, image, shape, channels):
        reshape_size = shape.append(channels)
        return np.reshape(image, reshape_size)

    # Accepts an array and number of categories, returns the one hot of the array
    def get_one_hot(self, array, categories):
        array = np.array(array, dtype=np.int32)
        b = np.zeros((array.size, categories))
        b[np.arange(array.size), array] = 1
        return b


def load_dataset(directory):
    dataset = GenderDataset()
    dataset.load_from_images(directory, [128, 128], 3)
    return dataset


def run():
    def weight_variable(shape):
        # Avoids symmetry when training, and prevents 0 gradients
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        # Avoids dead neurons
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    shape = 128
    channels = 3
    flat_in = shape * shape * channels
    flat_out = (int)((shape * shape * channels) / 16)
    out_size = 2
    # Input and Expect Output
    x = tf.placeholder(tf.float32, shape=[None, flat_in])
    y_ = tf.placeholder(tf.float32, shape=[None, out_size])

    W_conv1 = weight_variable([5, 5, 3, 32 * 3])
    b_conv1 = bias_variable([32 * 3])

    x_image = tf.reshape(x, [-1, shape, shape, channels])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)  # 64 59

    W_conv2 = weight_variable([5, 5, 32 * 3, 64 * 3])
    b_conv2 = bias_variable([64 * 3])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)  # 32 59

    W_fc1 = weight_variable([flat_out * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, flat_out * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, out_size])
    b_fc2 = bias_variable([out_size])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # Loss
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

    train_step = tf.train.AdamOptimizer(.0001).minimize(cross_entropy)


    max_y_conv = tf.argmax(y_conv, 1, name="convolution_max")
    max_y_ = tf.argmax(y_, 1, name="expected_max")

    correct_prediction = tf.equal(max_y_conv, max_y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Initialize Variables
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        dataset = load_dataset("Training/images")
        epochs = 100
        epoch_mod = 5
        batch_size = dataset.batch_size

        eval_dataset = dataset.get_full_batch()
        training_dataset = dataset.repeat(epochs).shuffle()

        epoch_range = range(epochs)
        for epoch in epoch_range:
            batch = training_dataset.get_batch_from_epoch(batch_size,epoch)
            if epoch % epoch_mod == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                print("step %d, training accuracy %g" % (epoch, train_accuracy))
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

        print(sess.run(accuracy,feed_dict={x: eval_dataset[0], y_: eval_dataset[1], keep_prob: 1.0}))
