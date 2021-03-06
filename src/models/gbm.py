# -*- coding: utf-8 -*-
# @Author: denglei
# @Date:   2018-05-21 16:36:51
# @Last Modified by:   denis
# @Last Modified time: 2018-05-23 15:16:27


import os
from . import BaseClassifier
from lightgbm import LGBMClassifier


class GBMClassifier(BaseClassifier):

	def __init__(self, opt):
		super().__init__(opt)
		self.clf_name = 'GBMClassifier'
		self.clf = LGBMClassifier(
			n_estimators       = opt.get('n_estimators', 1000),
			learning_rate      = opt.get('lr_rate', 0.07),
			num_leaves         = opt.get('num_leaves', 63),
			max_depth          = opt.get('max_depth', 7),
			colsample_bytree   = opt.get('cols_frac', 0.7),
			subsample          = opt.get('rows_frac', 0.7),
			reg_alpha          = opt.get('reg_alpha', 0.1),
			reg_lambda         = opt.get('reg_lambda', 0.1),
			min_split_gain     = opt.get('min_split_gain', 0.01),
			min_child_weight   = opt.get('min_child_weight', 2),
			max_bin            = opt.get('max_bin', 255),
			scale_pos_weight   = opt.get('scale_pos_weight', 1),    # model that not added before good_v3_p2
			random_state       = opt.get('random_state', 18520),
			n_jobs             = opt.get('n_jobs', 2),
			data_random_seed      = 1000,
			feature_fraction_seed = 1001,
			bagging_seed          = 1002
		)
		self.es_stop_num = opt.get('early_stopping_round', 20)

	def fit(self, train_set, valid_set=None):
		"""
		训练模型

		@params:

		train_set: (train_x, train_y), train_x 应该是pandas 并且里面指定好category变量
		valid_set: 同train_set

		"""
		self.clf.fit(
			train_set[0], train_set[1], 
			eval_set=[valid_set], eval_names='valid',
			eval_metric=self.metric, 
			early_stopping_rounds=self.es_stop_num
			)

	def get_feat_imp(self):
		return self.clf.feature_importances_

	def predict_proba(self, x):
		"""
		输出概率正样本的
		"""
		return self.clf.predict_proba(x, num_iteration=self.clf.best_iteration_)[:, 1]

	def save_model(self, model_dir='.', name=None, prefix=None):
		"""
		输出模型到文件中
		"""
		if name is None:
			name = self.get_model_name(prefix)

		name = os.path.join(model_dir, name)

		self.clf.booster_.save_model(name)