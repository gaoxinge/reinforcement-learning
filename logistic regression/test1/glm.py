# -*- coding: utf-8 -*-
import atexit
import tensorflow as tf

# make sure that session close
global_session = []

def _python_exit():
    for session in global_session:
        session.close()

atexit.register(_python_exit)


# logistic regression
def _enumerate(xs, ys):
    zs = []
    m = min(len(xs), len(ys))
    for index in range(m):
        zs.append((xs[index], ys[index]))
    return zs


def _func(prob):
    if prob >= 0.5:
        return 1
    else:
        return 0


class LogisticRegression:

    def __init__(self, feature_num, learning_rate=0.001, episode=100):
        self.feature_num = feature_num
        self.learning_rate = learning_rate
        self.episode = episode

        self._build()
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        global_session.append(self.session)

    def _build(self):
        self.x = tf.placeholder(tf.float32, [None, self.feature_num])
        self.y = tf.placeholder(tf.float32, [None, 1])
        self.w = tf.Variable(tf.zeros([self.feature_num, 1]))
        self.b = tf.Variable(tf.zeros([1]))
        self.y_ = tf.nn.sigmoid(tf.matmul(self.x, self.w) + self.b)

        self.loss = -tf.reduce_mean(tf.reduce_sum(
            self.y * tf.log(self.y_) + (1 - self.y) * tf.log(1 - self.y_),
            reduction_indices=1
        ))
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

    def fit(self, train_x, train_y):
        for _ in range(self.episode):
            for x, y in _enumerate(train_x, train_y):
                self.session.run(self.optimizer, feed_dict={self.x: [x], self.y: [[y]]})

    def predict(self, predict_x):
        predict_y = []
        for x in predict_x:
            prob = self.session.run(self.y_, feed_dict={self.x: [x]})[0][0]
            class_ = _func(prob)
            predict_y.append(class_)
        return predict_y
