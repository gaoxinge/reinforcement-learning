# -*- coding: utf-8 -*-


class LinearModel(object):

    def __init__(self, feat_n, get_feature):
        self.feat_n = feat_n
        self.get_feature = get_feature
        self.w = [0.0 for _ in range(self.feat_n)]

    def __call__(self, *args):
        feature = self.get_feature(*args)
        return sum([self.w[i] * feature[i] for i in range(self.feat_n)])
        
    def update(self, error, *args):
        feature = self.get_feature(*args)
        for i in range(self.feat_n):
            self.w[i] += error * feature[i]
