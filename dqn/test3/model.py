# -*- coding: utf-8 -*-
import tensorflow as tf


def _add_layer(x, input_size, output_size, activation_function=None):
    w = tf.Variable(tf.zeros([input_size, output_size]))
    b = tf.Variable(tf.zeros([1, output_size]))
    y = tf.matmul(x, w) + b
    if activation_function is None:
        return y
    else:
        return activation_function(y)


class Model:

    def __init__(self, input_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.x = tf.placeholder(tf.float32, [None, input_size])
        self.y = tf.placeholder(tf.float32, [None, output_size])

        self._add()
        self._compile()

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def _add(self):
        layer = _add_layer(self.x, self.input_size, 10, activation_function=tf.nn.relu)
        self.y_ = _add_layer(layer, 10, self.output_size, activation_function=tf.nn.relu)

    def _compile(self):
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.y - self.y_), reduction_indices=1))
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

    def fit(self, x, y):
        self.session.run(self.optimizer, feed_dict={self.x: x, self.y: y})

    def predict(self, x):
        return self.session.run(self.y_, feed_dict={self.x: x})