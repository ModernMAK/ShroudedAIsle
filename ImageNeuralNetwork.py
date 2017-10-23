from GenderAI import GenderDataset
import tensorflow as tf

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
        self.logits = graph.logits_from_inn(self, params=params)
        self.loss = graph.loss_from_inn(self, self.logits)
        self.training = graph.training(self.loss, params=params)
        self.evaluation = graph.evaluation_from_inn(self, self.logits)