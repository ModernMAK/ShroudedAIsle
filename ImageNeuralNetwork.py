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

        self.t_inference = None
        self.t_loss = None
        self.t_training = None
        self.t_evaluation = None
        self.t_predict = None

    def feed_dict(self, images=None, labels=None, probability=None):
        feed_dict = {}
        if images is not None:
            feed_dict[self.images] = images
        if labels is not None:
            feed_dict[self.labels] = labels
        if probability is not None:
            feed_dict[self.probability] = probability
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
            "inference": get_tensor_name(self.t_inference),
            "loss": get_tensor_name(self.t_loss),
            "training": get_tensor_name(self.t_training),
            "evaluation": get_tensor_name(self.t_evaluation),
            "predict": get_tensor_name(self.t_predict)
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
