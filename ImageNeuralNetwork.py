import tensorflow as tf


class ImageNeuralNetwork:
    def __init__(self, image_shape, channels, categories, graph=None):
        self.shape = image_shape
        self.channels = channels
        self.categories = categories
        self.flat_shape = image_shape[0] * image_shape[1] * channels
        if graph is None:
            graph = tf.get_default_graph()
        self.graph = graph
        with self.graph.as_default():
            with tf.name_scope("inputs"):
                self.probability = tf.placeholder(tf.float32)
                self.images = tf.placeholder(tf.float32, shape=[None, self.flat_shape], name="images")
                self.labels = tf.placeholder(tf.float32, shape=[None, categories], name="labels")
        self.inference = None
        self.loss = None
        self.training = None
        self.evaluation = None

    def feed_dict(self, images, labels, probability):
        feed_dict = {
            self.images: images,
            self.labels: labels,
            self.probability: probability
        }
        return feed_dict

    def get_name_dict(self):
        def get_tensor_name(tensor):
            if tensor is None:
                return None
            return tensor.name

        return {
            "images": get_tensor_name(self.images),
            "labels": get_tensor_name(self.labels),
            "probability": get_tensor_name(self.probability),
            "inference": get_tensor_name(self.inference),
            "loss": get_tensor_name(self.loss),
            "training": get_tensor_name(self.training),
            "evaluation": get_tensor_name(self.evaluation)
        }

    def get_tensor_dict(self):
        tensor_dict = {}
        for key, value in self.get_name_dict().items():
            if value is None:
                tensor_dict[key] = value
            else:
                try:
                    tensor_dict[key] = self.graph.get_tensor_by_name(value)
                except ValueError:
                    tensor_dict[key] = self.graph.get_operation_by_name(value)
        return tensor_dict
