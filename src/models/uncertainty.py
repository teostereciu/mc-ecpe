import math
from tqdm import trange
import tensorflow as tf
import numpy as np

def make_batches(iter_x, iter_y, batch_size=32):
    l = len(iter_x)
 
    for ndx in range(0, l, batch_size):
        x = iter_x[ndx:min(ndx + batch_size, l)]
        y = iter_y[ndx:min(ndx + batch_size, l)]
        
        yield x, y

def find_rbf_layer(model):
    rbf_layers = []

    for layer in model.layers:
        if type(layer) is RBFClassifier:
            rbf_layers.append(layer)

    if len(rbf_layers) == 1:
        return rbf_layers[0]
    
    raise ValueError("Multiple RBF layers detected, current training loop assumes only one RBF layer, cannot proceed. You can use your own custom training loop")

def duq_training_loop(model, input_feature_model, x_train, y_train, epochs=10, batch_size=32, validation_data=None, penalty_type="two-sided", lambda_coeff=0.5, callbacks=None):
    rbf_layer = find_rbf_layer(model)
    num_batches = math.ceil(x_train.shape[0] / batch_size)
    factor = 0.5
    
    for epoch in range(epochs):
        t = trange(num_batches, desc='Epoch {} / {}'.format(epoch, epochs))

        metric_loss_values = {}
        metric_names = model.metrics_names

        for metric_name in metric_names:
            metric_loss_values[metric_name] = []

        for i, (x_batch_train, y_batch_train) in zip(t, make_batches(x_train, y_train)):
            loss_metrics = model.train_on_batch(x_batch_train, y_batch_train)
            
            x_batch_rbf = input_feature_model.predict(x_batch_train)
            rbf_layer.update_centroids(x_batch_rbf, y_batch_train)

            metric_means = []

            for name, value in zip(metric_names, loss_metrics):
                metric_loss_values[name].append(value)
                metric_means.append(np.mean(metric_loss_values[name]))
 
            desc = " ".join(["{}: {:.3f}".format(name, value) for name, value in zip(metric_names, metric_means)])
            t.set_description('Epoch {} / {} - '.format(epoch + 1, epochs) + desc)
            t.refresh()

        if validation_data is not None:
            x_val, y_val = validation_data
            val_loss_metrics = model.evaluate(x_val, y_val, batch_size=batch_size, verbose=0)

            desc = " ".join(["{}: {:.3f}".format(name, value) for name, value in zip(metric_names, val_loss_metrics)])
            print("Validation metrics: {}".format(desc))


def add_gradient_penalty(model, lambda_coeff=0.5, penalty_type="two-sided"):
    def gradient_penalty_loss(y_true, y_pred):
        with tf.GradientTape() as tape:
            tape.watch(model.input)
            output = model(model.input)
            term = tape.gradient(tf.reduce_sum(output, axis=1), model.input)
            term = tf.square(term)

            if penalty_type == "two-sided":
                penalty = tf.square(term - 1)
            elif penalty_type == "one-sided":
                penalty = tf.maximum(0, term - 1)
            else:
                raise ValueError("Invalid penalty type {}, valid values are [one-sided, two-sided]".format(penalty_type))

            penalty = lambda_coeff * penalty
        return penalty
    model.add_loss(gradient_penalty_loss)


def add_l2_regularization(model, l2_strength=1e-4):
    for layer in model.layers:
        for tw in layer.trainable_weights:
            regularization_loss = tf.keras.regularizers.L2(l2_strength)(tw)
            model.add_loss(regularization_loss)

class RBFClassifier(tf.keras.layers.Layer):
    """
    Implementation of direct uncertainty quantification (DUQ)
    Reference: 
        Uncertainty Estimation Using a Single Deep Deterministic Neural Network
        Amersfoort et al. ICML 2020.

    """
    def __init__(self, num_classes, length_scale, centroid_dims=2, kernel_initializer="he_normal", centroids_initializer="uniform",
                 gamma=0.99, trainable_centroids=False, **kwargs):
        tf.keras.layers.Layer.__init__(self, **kwargs)        
        self.num_classes = num_classes
        self.centroid_dims = centroid_dims
        self.length_scale = length_scale
        self.gamma = gamma
        self.trainable_centroids = trainable_centroids

        self.kernel_initializer = kernel_initializer
        self.centroids_initializer = centroids_initializer

    def build(self, input_shape):
        in_features = input_shape[-1]

        centroid_init = self.centroids_initializer if self.trainable_centroids else "zeros"

        self.centroids = self.add_weight(name="centroids", shape=(self.centroid_dims, self.num_classes), dtype="float32", trainable=self.trainable_centroids, initializer=centroid_init)
        self.kernels = self.add_weight(name="kernels", shape=(self.centroid_dims, self.num_classes, in_features), initializer=self.kernel_initializer)

        self.m = np.zeros(shape=(self.centroid_dims, self.num_classes))
        self.n = np.ones(shape=(self.num_classes))

    def compute_output_shape(self, input_shape):
        return [(None, self.num_classes)]

    def call(self, inputs, training=None):
        z = tf.einsum("ij,mnj->imn", inputs, self.kernels)
        out = self.rbf(z)

        return out

    def rbf(self, z):
        z = z - self.centroids
        z = tf.keras.backend.mean(tf.keras.backend.square(z), axis=1) / (2.0 * self.length_scale ** 2)
        z = tf.keras.backend.exp(-z)

        return z
    
    def update_centroids(self, inputs, targets):
        kernels = tf.keras.backend.get_value(self.kernels)
        z = np.einsum("ij,mnj->imn", inputs, kernels)

        # Here we assume that targets is one-hot encoded.
        class_counts = np.sum(targets, axis=0)
        centroids_sum = np.einsum("ijk,ik->jk", z, targets)

        self.n = self.n * self.gamma + (1 - self.gamma) * class_counts
        self.m = self.m * self.gamma + (1 - self.gamma) * centroids_sum

        tf.keras.backend.set_value(self.centroids, self.m / self.n)

    def get_config(self):
        cfg = tf.keras.layers.Layer.get_config(self)
        cfg["num_classes"] = self.num_classes
        cfg["length_scale"] = self.length_scale
        cfg["centroid_dims"] = self.centroid_dims
        cfg["kernel_initializer"] = self.kernel_initializer
        cfg["gamma"] = self.gamma

        return cfg