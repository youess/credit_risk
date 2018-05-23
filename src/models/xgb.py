# -*- coding: utf-8 -*-
# @Author: denis
# @Date:   2018-05-23 15:42:59
# @Last Modified by:   denis
# @Last Modified time: 2018-05-23 18:03:41



import xgboost as xgb
from . import BaseClassifier


class XGB_Classifier(BaseClassifier):

	def __init__(self, opt):
		super().__init__(opt)
		self.clf_name = 'XGB_Classifier'
		self.clf = xgb.XGBClassifier(
			max_depth        = opt.get('max_depth', 8), 
			learning_rate    = opt.get('learning_rate', 0.1),
			n_estimators     = opt.get('n_estimators', 5000),
			max_leaf_nodes   = opt.get('max_leaf_nodes', 127),
			subsample        = opt.get('subsample', 0.7),
			colsample_bytree = opt.get('colsample_bytree', 0.7),
			gamma            = opt.get('gamma', 0.1),
			min_child_weight = opt.get('min_child_weight', 4),
			reg_alpha        = opt.get('reg_alpha', 0.1),
			reg_lambda       = opt.get('reg_lambda', 0.1),
			scale_pos_weight = opt.get('scale_pos_weight', 10),
			n_jobs           = opt.get('n_jobs', 2),
			seed             = opt.get('seed', 123),
			random_state     = opt.get('random_state', 18520)
			)
		self.es_stop_num = opt.get('early_stopping_round', 50)

	def fit(self, train_set, valid_set=None):
		self.clf.fit(train_set[0], train_set[1],
			eval_set=[valid_set],
			eval_metric='auc',
			early_stopping_rounds=self.es_stop_num)

	def predict_proba(self, x):
		bst_iter = self.clf.get_booster().best_ntree_limit
		return self.clf.predict_proba(x, ntree_limit=bst_iter)[:, 1]

	def get_feat_imp(self):
		score_dict = self.clf.get_booster().get_fscore()
		feat_list = self.clf.get_booster().feature_names
		feat_score = [0 for _ in range(len(feat_list))]
		for i, f in enumerate(feat_list):
			feat_score[i] = score_dict[f]
		return feat_score
