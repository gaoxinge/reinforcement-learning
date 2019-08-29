# -*- coding: utf-8 -*-
import tensorflow as tf


def _add_layer(x, input_size, output_size, activation_function=None):
    w = tf.Variable(tf.random_normal([input_size, output_size])) # don't initialize zero
    b = tf.Variable(tf.zeros(shape=[1, output_size]) + 0.1)      # don't initialize zero
    y = tf.matmul(x, w) + b
    if activation_function is None:
        return y
    else:
        return activation_function(y)


class Model:

    def __init__(self, state_n, act_n, learning_rate):
        self.state_n = state_n
        self.act_n = act_n
        self.learning_rate = learning_rate
        self.state = tf.placeholder(tf.float32, [None, state_n])
        self.action = tf.placeholder(tf.int32)
        self.total_reward = tf.placeholder(tf.float32)

        self._add()
        self._compile()

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def _add(self):
        layer = _add_layer(self.state, self.state_n, 24) # relu sucks
        layer = _add_layer(layer, 24, 24)                # relu sucks
        self.probs = _add_layer(layer, 24, self.act_n, activation_function=tf.nn.softmax)
        self.probs = tf.squeeze(self.probs)

    def _compile(self):
        self.prob = tf.gather(self.probs, self.action)
        self.loss = -tf.log(self.prob) * self.total_reward
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
    
    def predict(self, state):
        feed_dict = {self.state: state}
        return self.session.run(self.probs, feed_dict=feed_dict)

    def fit(self, state, action, total_reward):
        feed_dict = {self.state: state, self.action: action, self.total_reward: total_reward}
        self.session.run(self.optimizer, feed_dict=feed_dict)
