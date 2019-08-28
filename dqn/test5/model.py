# -*- coding: utf-8 -*-
import tensorflow as tf


def _add_layer(x, input_size, output_size, activation_function=None):
    w = tf.Variable(tf.random_normal([input_size, output_size])) # don't initialize zero
    b = tf.Variable(tf.zeros([1, output_size]) + 0.1)            # don't initialize zero
    y = tf.matmul(x, w) + b
    if activation_function is None:
        return y
    else:
        return activation_function(y)


class Model:

    def __init__(self, input_size, output_size, learning_rate):
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
        layer = _add_layer(self.x, self.input_size, 24, activation_function=tf.nn.relu)
        layer = _add_layer(layer, 24, 24,  activation_function=tf.nn.relu)
        self.y_ = _add_layer(layer, 24, self.output_size)

    def _compile(self):
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.y - self.y_), reduction_indices=1))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def fit(self, x, y):
        self.session.run(self.optimizer, feed_dict={self.x: x, self.y: y})

    def predict(self, x):
        return self.session.run(self.y_, feed_dict={self.x: x})
