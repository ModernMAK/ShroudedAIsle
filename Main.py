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

from GenderBrain import ImageGraph, ImageNeuralNetwork, GenderDataset, DisposableGenderDataset
import tensorflow as tf


def load_dataset_from_directory(directory):
    dataset = GenderDataset()
    dataset.load_from_images(directory, [128, 128], 3)
    return dataset


def load_dataset_from_file(file, mode):
    dataset = GenderDataset()
    dataset.load_from_file(file, mode=mode)
    return dataset


def run():
    raw_dataset = load_dataset_from_directory("Training/images")
    training_dataset = DisposableGenderDataset(raw_dataset).repeat(1000, True)
    brain = ImageNeuralNetwork([128, 128], 3, 2)
    brain.build_graph(ImageGraph())

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(1000):
            batch = training_dataset.get_next_batch(16)
            if epoch % 10 == 0:
                training_accuraccy = brain.evaluation.eval(feed_dict=brain.feed_dict(batch[0], batch[1], 1))
                print(" step %d, accuraccy: %f " % (epoch, training_accuraccy))
            brain.training.run(feed_dict=brain.feed_dict(batch[0], batch[1], 0.5))


if __name__ == "__main__":
    print("RUNNING")
    # run()
    from DataCollector import collect_gender_data_from_game
    collect_gender_data_from_game()
#
# def run():
#     # Params is a dict with
#     # params['conv1'] = int
#     # params['conv2'] = int
#     # params['fcl'] = int
#     # params['shape'] = [int,int] (REQ)
#     # params['channels] = int (REQ)
#     # params['genders'] = int
#     # params['keep'] = float
#
#     params = {
#         "conv1": 32,
#         "conv2": 64,
#         "fcl": 1024,
#         "shape": [128, 128],
#         "channels": 3,
#         "genders": 2,
#         "keep": 0.5
#     }
#     batchsize = 16
#     learning_rate = 0.1
#     logdir = "/graph"
#     max_steps = 100
#
#     from six.moves import xrange
#
#     def fill_dict(disposable_dataset, images_pl, labels_pl):
#         batch = disposable_dataset.get_next_batch(batchsize)
#         feed_dict = {
#             images_pl: batch[0],
#             labels_pl: batch[1]
#         }
#         return feed_dict
#
#     def do_eval(sess, eval_correct, images_placeholder, labels_placeholder, dataset):
#         correct_count = 0
#         steps_per_epoch = dataset.size // batchsize
#         num_examples = steps_per_epoch * batchsize
#         for step in xrange(steps_per_epoch):
#             feed_dict = fill_dict(dataset, images_placeholder, labels_placeholder)
#             correct_count += sess.run(eval_correct, feed_dict=feed_dict)
#         precision = float(correct_count) / num_examples
#         print(' Tested: %d Correct: %d Precision @ 1: %0.04f' % num_examples, correct_count, precision)
#
#     def run_training():
#         raw_dataset = load_dataset_from_directory("Training/images")
#         dataset = DisposableGenderDataset(raw_dataset)
#         brain = GenderBrain()
#         with tf.Graph().as_default():
#             images_placeholder, labels_placeholder = brain.inputs(params)
#             logits = brain.inference(images_placeholder, params)
#             loss = brain.loss(logits, labels_placeholder)
#             train_op = brain.training(loss, learning_rate)
#             eval_correct = brain.evaluation(logits, labels_placeholder)
#             summary = tf.summary.merge_all()
#
#             init = tf.global_variables_initializer()
#             saver = tf.train.Saver()
#             with tf.Session() as sess:
#                 with tf.summary.FileWriter(logdir, sess.graph) as summary_writer:
#                     sess.run(init)
#                     for step in xrange(max_steps):
#                         start_time = time.time()
#                         feed_dict = fill_dict(
#                             dataset,
#                             images_placeholder, labels_placeholder, probability,
#                             0.5
#                         )
#                         _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
#
#                         duration = time.time() - start_time
#
#                         if step % 100 == 0:
#                             print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
#
#                             summary_str = sess.run(summary, feed_dict=feed_dict)
#                             summary_writer.add_summary(summary_str, step)
#                             summary_writer.flush()
#
#                             # We could do this, but I don't have a validation, or test set For future me, a validation
#                             #  set is used to check the training set and measure progress For future me, a test set
#                             # is used to check for over fitting (a set never run through the learning,
#                             # if it's producing significant losses, the graph is overfit)
#
#                             # if (step + 1) % 1000 == 0 or (step + 1) == max_steps:
#                             #     checkpoint_file = os.path.join(logdir,'model.ckpt')
#                             #     saver.save(sess,checkpoint_file,global_step=step)
#                             #     print('Training Data Eval:')
#                             #     do_eval(sess,images_placeholder,labels_placeholder,train_dataset)
#                             #     print('Validation Data Eval:')
#                             #     do_eval(sess,images_placeholder,labels_placeholder,train_dataset)
#                             #     print('Training Data Eval:')
#                             #     do_eval(sess,images_placeholder,labels_placeholder,train_dataset)
#
#     run_training()
