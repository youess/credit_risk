# -*- coding: utf-8 -*-
# @Author: denglei
# @Date:   2018-05-21 16:36:21
# @Last Modified by:   denglei
# @Last Modified time: 2018-05-21 16:59:03


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