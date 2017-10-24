import tensorflow as tf
from os.path import join
from ImageNeuralNetwork import ImageNeuralNetwork
from ImageGraph import build_convolutional_neural_network


# This class is a brain which will run gender recognition
# I divide the graph into "brains" that solve individual tasks, later I hope they will be able
class GenderBrain(ImageNeuralNetwork):
    def __init__(self, graph_dir, params={}):
        super().__init__([128, 128], 3, 2, tf.Graph())
        self.graph_dir = graph_dir
        self.graph_tensor_dict = {}
        build_convolutional_neural_network(self, "gender_cnn", params)


    def save(self, session, step=None):
        # with session.graph.as_default():
        saver = tf.train.Saver()
        file_path = join(self.graph_dir, "gender_cnn")
        if step is None:
            saver.save(session, file_path)
        else:
            saver.save(session, file_path, global_step=step)

    def load(self, session):
        loader = tf.train.Saver()
        file_path = self.graph_dir
        ckpt = tf.train.latest_checkpoint(file_path)
        print(str(ckpt))
        loader.restore(session, ckpt)

    def session(self):
        with self.graph.as_default() as g:
            return tf.Session(graph=g)

    # Assumes a disposable data set was passed in
    def train(self, dataset, epochs, batch_size, *, print_every_nth_step=None, save_every_nth_step=None):
        print("running training")
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

                    sess.run(self.t_training, feed_dict=self.feed_dict(batch[0], batch[1], 0.5))

                    if print_every_nth_step is not None and step % print_every_nth_step == 0:
                        training_accuraccy = self.t_evaluation.eval(feed_dict=self.feed_dict(batch[0], batch[1], 1))
                        print(" step %d (%d.%d), accuraccy: %f " % (step, epoch, epoch_step, training_accuraccy))

                    if save_every_nth_step is not None and step % save_every_nth_step == 0:
                        self.save(sess, step)
                        print(" step %d saved " % step)
            self.save(sess)  # training.run(feed_dict=feed_dict_train)

    #NOTE this returns the full dataset
    def predict(self, dataset, batch_size):
        print("running prediction")
        steps_per_epoch = dataset.get_iterations_given_batch_size(batch_size)
        predictions = []
        with self.session() as sess:
            sess.run(tf.global_variables_initializer())
            self.load(sess)
            for epoch_step in range(steps_per_epoch):
                batch = dataset.get_next_batch(batch_size)
                prediction = self.t_predict.eval(feed_dict=self.feed_dict(images=batch[0], probability=1.0))
                predictions.extend(prediction)
        return predictions
