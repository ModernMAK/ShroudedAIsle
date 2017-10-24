# # This is a fun little idea I cooked up while playing the Shrouded Isle
# # A game which is rather simple to play (Select Court, Use members of the Court, Find a Sinner, Execute, Repeat)
# # but is rather complicated to complete (Manage Virtues, Find Vices, Maintain Relations, Complete Dreams)
# # Because its a point in click game, I feel it is "pretty simple" to hack up some rudimentary AI to play it.
# # That being said, I have no idea how to use Tensorflow, nor do I have access to an API to interact with the game.

from ImageDataset import DisposableImageDataSet, ImageGenderDataSet, ImageDataSet
from os.path import join, basename
from os import getcwd, rename
from GenderBrain import GenderBrain
from time import sleep


def load_dataset_from_directory(directory, is_prediction=False):
    if is_prediction:
        dataset = ImageDataSet()
    else:
        dataset = ImageGenderDataSet()
    dataset.load_from_images(directory, [128, 128], 3)
    return dataset


def load_dataset_from_file(file, mode):
    dataset = ImageGenderDataSet()
    dataset.load_from_file(file, mode=mode)
    return dataset


def run_training(epochs=100, batch_size=32):
    raw_dataset = load_dataset_from_directory("Data/Training")
    training_dataset = DisposableImageDataSet(raw_dataset).repeat(epochs, shuffle_on_repeat=True)

    params = {
        "learning_rate": 0.001
    }
    brain = GenderBrain("graph/gender", params=params)
    # brain.build_graph()
    brain.train(training_dataset, epochs, batch_size, print_every_nth_step=10, save_every_nth_step=10)


def run_prediction(batch_size=32):
    raw_dataset = load_dataset_from_directory("Data\Predict\\Unsorted", True)
    predict_dataset = DisposableImageDataSet(raw_dataset)

    def get_category(pair):
        if pair[0] < pair[1]:
            return 1
        elif pair[0] > pair[1]:
            return 0
        else:
            raise Exception("Equal!?")

    brain = GenderBrain("graph/gender")
    # brain.build_graph()
    predictions = brain.predict(predict_dataset, batch_size)
    for i in range(predict_dataset.size):
        out_dir = "Data\Predict\\Unsorted"
        pred = get_category(predictions[i])
        if pred == 0:
            out_dir = "Data\Predict\Male"
        elif pred == 1:
            out_dir = "Data\Predict\Female"
        else:
            print("Uh... Houstin, there's a problem...")

        file_path = predict_dataset[1][i]
        file_name = basename(file_path)
        rename(file_path, join(join(getcwd(), out_dir), file_name))
        sleep(0.01)


#
# def run(epochs=100, batch_size=64, shuffle_on_repeat=True):
#     def try_save(sess, model, step, dict):
#         saver = tf.train.Saver()
#         saver.save(sess, model, global_step=step)
#         # with open(model + ".dict", 'w') as file:
#         #     for key, value in dict.items():
#         #         file.write(key)
#         #         file.write(value)
#
#     def try_restore(sess, model, step=0):
#         # model_dir = dirname(model)
#         file_name = "%s-%d.meta" % (model, step)
#         loader = tf.train.import_meta_graph(file_name)
#         # latest_checkpoint = tf.train.latest_checkpoint(model_dir)
#         # loader = tf.train.import_meta_graph(model)
#         # checkpoint = tf.train.latest_checkpoint(model)
#         loader.restore(sess, file_name)
#
#
#         # dict = {}
#         # with open(model + ".dict", 'r') as file:
#         #     prev = None
#         #     for line in file:
#         #         if prev is not None:
#         #             dict[prev] = line
#         #             prev = None
#         #         else:
#         #             prev = line
#         # return dict
#
#     raw_dataset = load_dataset_from_directory("Data/Training")
#     training_dataset = DisposableImageDataSet(raw_dataset).repeat(epochs, shuffle_on_repeat)
#     brain = ImageNeuralNetwork([128, 128], 3, 2)
#     params = {"learning_rate": 0.0001}
#     brain.build_graph(ImageGraph(), params=params)
#     iterations = training_dataset.get_iterations_given_batch_size(batch_size)
#     print(training_dataset.size, "=>", iterations)
#
#     init = tf.global_variables_initializer()
#     checkpoint = "graph/checkpoints/gender"
#     dict = brain.get_name_dict()
#     feed_dict_train = {}
#     feed_dict_eval = {}
#     training = None
#     evaluation = None
#     with tf.Session() as sess:
#         def run_eval():
#             if epoch % 2 == 0 and i % 10 == 0:
#                 try_save(sess, checkpoint, epoch * iterations + i, dict)
#                 training_accuraccy = evaluation.eval(feed_dict=feed_dict_eval)
#                 print(" step %d.%d, accuraccy: %f " % (epoch, i, training_accuraccy))
#
#         def run_train():
#             training.run(feed_dict=feed_dict_train)
#
#         try:
#             dict = try_restore(sess, checkpoint)
#             graph = tf.get_default_graph()
#             training = graph.get_tensor_by_name(dict["training"])
#             evaluation = graph.get_tensor_by_name(dict["evaluation"])
#         except Exception as e:
#             sess.run(init)
#             dict = brain.get_name_dict()
#             training = brain.training
#             evaluation = brain.evaluation
#             print(str(e))
#
#         for epoch in range(epochs):
#             training_dataset.reset()
#             for i in range(iterations):
#                 batch = training_dataset.get_next_batch(batch_size)
#                 feed_dict_eval = brain.feed_dict(batch[0], batch[1], 1)
#                 feed_dict_train = brain.feed_dict(batch[0], batch[1], 0.5)
#                 run_eval()
#                 run_train()
#
#         try_save(sess, checkpoint, epochs * iterations)


def main():
    from DataCollector import collect_gender_data_from_game, cull_gender_data
    def print_menu():
        print("<-< Shrouded AIsle >->")
        print("1) Collect Gender Data")
        print("2) Cull Collected Data")
        print("3) Train Gender Recognition")
        print("4) Predict Gender")
        print("Q) Quit")
        print("----------------------")

    def parse_cmd(cmd):

        try:
            str_cmd = str(cmd)
        except:# We honestly dont care how it failed
            str_cmd = ""
        try:
            int_cmd = int(cmd)
        except:# We honestly dont care how it failed, we would only care if they both failed
            int_cmd = 0

        def collect():
            val = int(input("How many iterations?\n"))
            collect_gender_data_from_game(val, debounce=0.75, shuffle_colors=True)

        if int_cmd == 1 or str_cmd in ["co", "CO", "collect", "Collect", "COLLECT"]:
            collect()
        elif int_cmd == 2 or str_cmd in ["cu", "CU", "cull", "Cull", "CULL"]:
            cull_gender_data()
        elif int_cmd == 3 or str_cmd in ["T", "t", "train", "Train", "TRAIN"]:
            run_training()
        elif int_cmd == 4 or str_cmd in ["P", "p", "predict", "Predict", "PREDICT"]:
            run_prediction()
        elif str_cmd in ["q", "x", "X", "Q", "quit", "QUIT", "Quit"]:
            exit(0)
        else:
            print("I don't understand '%s'" % str(cmd))

    while True:
        print_menu()
        code = input()
        parse_cmd(code)


if __name__ == "__main__":
    print("RUNNING")
    main()
    # run()

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
#             inference = brain.inference(images_placeholder, params)
#             loss = brain.loss(inference, labels_placeholder)
#             train_op = brain.training(loss, learning_rate)
#             eval_correct = brain.evaluation(inference, labels_placeholder)
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
