# -*- coding: utf-8 -*-
# @Author: denis
# @Date:   2018-05-23 16:14:02
# @Last Modified by:   denis
# @Last Modified time: 2018-05-23 16:30:21


from catboost import CatBoostClassifier
from . import BaseClassifier


class CBClassifier(BaseClassifier):

	def __init__(self, opt):
		super().__init__(opt)
		self.clf_name = 'CBClassifier'
		self.es_stop_num = opt.get('early_stopping_round', 45)
		self.clf = xgb.CatBoostClassifier(
			iterations           = opt.get('iterations', 1000),
			learning_rate        = opt.get('learning_rate', 0.1),
			depth                = opt.get('depth', 7),
			l2_leaf_reg          = opt.get('l2_leaf_reg', 40),
			subsample            = opt.get('subsample', 0.7),
			scale_pos_weight     = opt.get('scale_pos_weight', 5),
			random_seed          = 18520,
			od_type              = 'Iter',
			od_wait              = self.es_stop_num,
			metric_period        = 50,
			eval_metric          = 'AUC',
			bootstrap_type       = 'Bernoulli',
			allow_writing_files  = False
			)
		

	def fit(self, train_set, valid_set=None):
		self.clf.fit(train_set[0], train_set[1],
			eval_set=valid_set,
			use_best_model=True)

	def predict_proba(self, x):
		return self.clf.predict_proba(x)[:, 1]

	def get_feat_imp(self):
		return self.clf.feature_importances_