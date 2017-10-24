import tensorflow as tf
from os.path import join
from ImageNeuralNetwork import ImageNeuralNetwork
from ImageGraph import build_convolutional_neural_network
from DataCollector import enforce_directory


# This class is a brain which will run gender recognition
# I divide the graph into "brains" that solve individual tasks, later I hope they will be able
class GenderBrain(ImageNeuralNetwork):
    def __init__(self, graph_dir, params={}):
        super().__init__([128, 128], 3, 2, tf.Graph())
        self.graph_dir = graph_dir
        self.graph_tensor_dict = {}
        build_convolutional_neural_network(self, "gender_cnn", params)

    # def get_graph_file(self, fname):
    #     return join(self.graph_dir, fname)
    #
    # def get_graph_dict(self):
    #     return self.get_graph_file('nodes.json')
    #
    # def save(self, session):
    #     def overwrite():
    #         from tensorflow.python.lib.io import file_io
    #         file_io.delete_recursively(self.graph_dir)
    #
    #     overwrite()
    #     with tf.saved_model.builder.SavedModelBuilder(self.graph_dir) as builder:
    #         builder.add_meta_graph_and_variables(
    #             session,
    #             [],
    #             signature_def_map=None,
    #             assets_collection=None
    #         )
    #         builder.save()
    #     with open(self.get_graph_dict(), 'w') as file:
    #         json.dump(self.graph_name_dict, file)
    #
    # # loads graph shape only
    # def load_graph(self):
    #     with tf.Graph().as_default() as g:
    #         with tf.Session(graph=g) as sess:
    #             self.load(sess)
    #             self.graph = g
    #
    # def load(self, session):
    #     tf.saved_model.loader.load(
    #         session,
    #         [],
    #         self.graph_dir
    #     )
    #     with open(self.get_graph_dict(), 'r') as file:
    #         self.graph_name_dict = json.load(file)
    #
    # def initialize_graph(self, session, params={}):
    #     try:
    #         self.load(session)
    #     except OSError as ose:
    #         print(str(ose), "\n", "Error! Resorting to global initialization...")
    #         init = tf.global_variables_initializer()
    #         self.initialize_new_graph(params)
    #         session.run(init)
    #     self.build_tensor_lookup()
    def save(self, session, step=None):
        # with session.graph.as_default():
        saver = tf.train.Saver()
        file_path = join(self.graph_dir, "gender_cnn")
        if step is None:
            saver.save(session, file_path)
        else:
            saver.save(session, file_path, global_step=step)
                # def overwrite():
                #     from tensorflow.python.lib.io import file_io
                #     if file_io.file_exists(self.graph_dir):
                #         try:
                #             file_io.delete_recursively(self.graph_dir)
                #         except Exception as e:
                #             # Try again
                #             print(str(e), "\n", "Trying again.")
                #             file_io.delete_recursively(self.graph_dir)
                #
                # overwrite()
                # builder = saved_model_builder.SavedModelBuilder(self.graph_dir)
                # builder.add_meta_graph_and_variables(
                #     session,
                #     [tag_constants.TRAINING],
                #     signature_def_map=None,
                #     assets_collection=None
                # )
                # builder.save()
                # # with open(self.get_graph_dict(), 'w') as file:
                # #     json.dump(self.graph_name_dict, file)

    def load(self, session):
        loader = tf.train.Saver()
        file_path = self.graph_dir
        # file_path = join(self.graph_dir, "gender_cnn")
        # enforce_directory(file_path)
        ckpt = tf.train.latest_checkpoint(file_path)
        print(str(ckpt))
        loader.restore(session, ckpt)
            # saved_model_loader.load(
            #     session,
            #     [tag_constants.TRAINING],
            #     self.graph_dir
            # )
            # with open(self.get_graph_dict(), 'r') as file:
            #     self.graph_name_dict = json.load(file)

    def session(self):
        with self.graph.as_default() as g:
            return tf.Session(graph=g)

    # Assumes a disposable data set was passed in
    def train(self, dataset, epochs, batch_size, *, print_every_nth_step=None, save_every_nth_step=None):
        steps_per_epoch = dataset.get_iterations_given_batch_size(batch_size)
        with self.session() as sess:
            sess.run(tf.global_variables_initializer())
            try:
                self.load(sess)
            except Exception as e:
                print(str(e))

            # tensors = self.get_tensor_dict()
            for epoch in range(epochs):
                dataset.reset()  # Reset the data set
                for epoch_step in range(steps_per_epoch):
                    step = epoch * steps_per_epoch + epoch_step
                    batch = dataset.get_next_batch(batch_size)

                    sess.run(self.training, feed_dict=self.feed_dict(batch[0], batch[1], 0.5))

                    if print_every_nth_step is not None and step % print_every_nth_step == 0:
                        training_accuraccy = self.evaluation.eval(feed_dict=self.feed_dict(batch[0], batch[1], 1))
                        print(" step %d (%d.%d), accuraccy: %f " % (step, epoch, epoch_step, training_accuraccy))

                    if save_every_nth_step is not None and step % save_every_nth_step == 0:
                        self.save(sess, step)
                        # print(" step %d saved " % step)
            self.save(sess)  # training.run(feed_dict=feed_dict_train)

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
