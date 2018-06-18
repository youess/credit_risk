# -*- coding: utf-8 -*-
# @Author: denis
# @Date:   2018-05-23 14:06:48
# @Last Modified by:   denis
# @Last Modified time: 2018-05-23 17:45:27


from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier
)
from . import BaseClassifier


class RFClassifier(BaseClassifier):
    def __init__(self, opt):
        super().__init__(opt)
        self.clf_name = 'RFClassifier'
        self.clf = RandomForestClassifier(
            n_estimators=opt.get('n_estimators', 200),
            max_depth=opt.get('max_depth', 7),
            min_samples_leaf=opt.get('min_samples_leaf', 10),
            max_leaf_nodes=opt.get('max_leaf_nodes', 63),
            min_samples_split=opt.get('min_samples_split', 2),
            class_weight=opt.get('class_weight', {0: 1, 1: 10}),
            random_state=opt.get('random_state', 18520),
            n_jobs=opt.get('n_jobs', 2)
        )

    def fit(self, train_set, valid_set=None):
        self.clf.fit(train_set[0], train_set[1])

    def predict_proba(self, x):
        return self.clf.predict_proba(x)[:, 1]

    def get_feat_imp(self):
        return self.clf.feature_importances_


class ETClassifier(BaseClassifier):
    def __init__(self, opt):
        super().__init__(opt)
        self.clf_name = 'ETClassifier'
        self.clf = ExtraTreesClassifier(
            n_estimators=opt.get('n_estimators', 200),
            max_depth=opt.get('max_depth', 7),
            min_samples_leaf=opt.get('min_samples_leaf', 10),
            max_leaf_nodes=opt.get('max_leaf_nodes', 63),
            min_samples_split=opt.get('min_samples_split', 2),
            bootstrap=opt.get('bootstrap', True),
            class_weight=opt.get('class_weight', {0: 1, 1: 10}),
            random_state=opt.get('random_state', 18520),
            n_jobs=opt.get('n_jobs', 2)
        )

    def fit(self, train_set, valid_set=None):
        self.clf.fit(train_set[0], train_set[1])

    def predict_proba(self, x):
        return self.clf.predict_proba(x)[:, 1]

    def get_feat_imp(self):
        return self.clf.feature_importances_
