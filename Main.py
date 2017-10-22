# # This is a fun little idea I cooked up while playing the Shrouded Isle
# # A game which is rather simple to play (Select Court, Use members of the Court, Find a Sinner, Execute, Repeat)
# # but is rather complicated to complete (Manage Virtues, Find Vices, Maintain Relations, Complete Dreams)
# # Because its a point in click game, I feel it is "pretty simple" to hack up some rudimentary AI to play it.
# # That being said, I have no idea how to use Tensorflow, nor do I have access to an API to interact with the game.
#
# # This means I have a few problems to solve
# # Input (Have to get a python library to emulate clicks)
# # Gathering Information without API (Have to use Image Recognition)
# # Decision making (unmentioned above, I need to come up with a hueristic to determine if the bot is making progress)
#
# # Lets look at the basics of what we have to gather, and how the AI must act to acquire this
#
# # ACHIEVING SATISFACTION
# # To learn satisfaction, on the home screen, we can find it by hovering over the name of the house/their satisfaction note
# # Once hovering, left of their name and satisfaction node, is their satisfaction value
# # ADDITIONALLY, we can use this method to check inquiries, which can be accessed elsewhere
#
# # ACHIEVING VIRTUE
# # To learn of the value of a virtue, we must hover over the virtue or the bar beneath
# # The value is right of the virtue name once Hovered
# # Additionally, to check if we are below the threshold, we have to check if the color is different OR if we are left of the bar
#
# # FIDNING SINNERS
# # There are many ways to find sinners, the primary being inquiries
#
# from WindowUtil import get_window_named, scan_window
# from time import sleep
# # print("Please Open \"The Shrouded Isle\"")
# # game = None
# # while game is None:
# #     print("Waiting...")
# #     sleep(5)
# #     game = get_window_named("The Shrouded Isle")
# # print("\"The Shrouded Isle\" was opened")
# #
# # scan = scan_window(game)
# # scan.save("Scan.png", "PNG")
# import ScanManip
# # ScanManip.idea()

#
# #  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# #
# #  Licensed under the Apache License, Version 2.0 (the "License");
# #  you may not use this file except in compliance with the License.
# #  You may obtain a copy of the License at
# #
# #   http://www.apache.org/licenses/LICENSE-2.0
# #
# #  Unless required by applicable law or agreed to in writing, software
# #  distributed under the License is distributed on an "AS IS" BASIS,
# #  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# #  See the License for the specific language governing permissions and
# #  limitations under the License.
# """Convolutional Neural Network Estimator for MNIST, built with tf.layers."""
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
#
# import numpy as np
# import tensorflow as tf
#
# tf.logging.set_verbosity(tf.logging.INFO)
#
#
# def cnn_model_fn(features, labels, mode):
#     """Model function for CNN."""
#     # Input Layer
#     # Reshape X to 4-D tensor: [batch_size, width, height, channels]
#     # MNIST images are 28x28 pixels, and have one color channel
#     input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
#
#     # Convolutional Layer #1
#     # Computes 32 features using a 5x5 filter with ReLU activation.
#     # Padding is added to preserve width and height.
#     # Input Tensor Shape: [batch_size, 28, 28, 1]
#     # Output Tensor Shape: [batch_size, 28, 28, 32]
#     conv1 = tf.layers.conv2d(
#         inputs=input_layer,
#         filters=32,
#         kernel_size=[5, 5],
#         padding="same",
#         activation=tf.nn.relu)
#
#     # Pooling Layer #1
#     # First max pooling layer with a 2x2 filter and stride of 2
#     # Input Tensor Shape: [batch_size, 28, 28, 32]
#     # Output Tensor Shape: [batch_size, 14, 14, 32]
#     pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
#
#     # Convolutional Layer #2
#     # Computes 64 features using a 5x5 filter.
#     # Padding is added to preserve width and height.
#     # Input Tensor Shape: [batch_size, 14, 14, 32]
#     # Output Tensor Shape: [batch_size, 14, 14, 64]
#     conv2 = tf.layers.conv2d(
#         inputs=pool1,
#         filters=64,
#         kernel_size=[5, 5],
#         padding="same",
#         activation=tf.nn.relu)
#
#     # Pooling Layer #2
#     # Second max pooling layer with a 2x2 filter and stride of 2
#     # Input Tensor Shape: [batch_size, 14, 14, 64]
#     # Output Tensor Shape: [batch_size, 7, 7, 64]
#     pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
#
#     # Flatten tensor into a get_full_batch of vectors
#     # Input Tensor Shape: [batch_size, 7, 7, 64]
#     # Output Tensor Shape: [batch_size, 7 * 7 * 64]
#     pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
#
#     # Dense Layer
#     # Densely connected layer with 1024 neurons
#     # Input Tensor Shape: [batch_size, 7 * 7 * 64]
#     # Output Tensor Shape: [batch_size, 1024]
#     dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
#
#     # Add dropout operation; 0.6 probability that element will be kept
#     dropout = tf.layers.dropout(
#         inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
#
#     # Logits layer
#     # Input Tensor Shape: [batch_size, 1024]
#     # Output Tensor Shape: [batch_size, 10]
#     logits = tf.layers.dense(inputs=dropout, units=10)
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
#     onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
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
# def main(unused_argv):
#     # Load training and eval data
#     print("Load Training/Eval")
#     mnist = tf.contrib.learn.datasets.load_dataset("mnist")
#     train_data = mnist.train.images  # Returns np.array
#     train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
#     print(mnist.train.images)
#     print(mnist.train.labels)
#
#     eval_data = mnist.test.images  # Returns np.array
#     eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
#
#     # Create the Estimator
#     print("Creating Estimator")
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
#         batch_size=100,
#         num_epochs=None,
#         shuffle=True)
#     mnist_classifier.train(
#         input_fn=train_input_fn,
#         steps=20000,
#         hooks=[logging_hook])
#
#     # Evaluate the model and print results
#     eval_input_fn = tf.estimator.inputs.numpy_input_fn(
#         x={"x": eval_data},
#         y=eval_labels,
#         num_epochs=1,
#         shuffle=False)
#     eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
#     print(eval_results)
#
#
# if __name__ == "__main__":
#     print("RUNNING")
#     tf.app.run()
#

import GenderAI
GenderAI.run()