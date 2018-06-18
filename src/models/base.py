# -*- coding: utf-8 -*-
# @Author: denglei
# @Date:   2018-05-21 16:36:21
# @Last Modified by:   denis
# @Last Modified time: 2018-05-23 14:11:19


from datetime import datetime


class BaseClassifier(object):
    def __init__(self, opt):
        self.clf_name = 'BaseClassifier'
        self.clf = None
        self.metric = 'auc'

    def fit(self, train_set, valid_set=None):
        pass

    def predict_proba(self, x):
        pass

    def get_feat_imp(self):
        pass

    def save_model(self):
        pass

    def get_model_name(self, prefix=None):
        tmt = datetime.now().strftime('%Y%m%d_%H%M')
        name = self.clf_name + '_' + tmt + '.txt'
        if prefix is not None:
            name = prefix + name
        return name
