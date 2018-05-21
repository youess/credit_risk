# -*- coding: utf-8 -*-
# @Author: denglei
# @Date:   2018-05-21 09:39:03
# @Last Modified by:   denglei
# @Last Modified time: 2018-05-21 18:23:43


import os
import gc
import os.path as op
import numpy as np
import feather
import pandas as pd
import fire
import models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from datetime import datetime


def cache_read(csv_file, cache_suffix=".feather", **kwargs):
	"""
	用pandas读取csv文件，查看是否具有默认的缓存文件，
	如果没有缓存文件，那么按照原始文件进行读取
	"""
	cache_file = csv_file + cache_suffix
	if op.exists(cache_file):
		data = feather.read_dataframe(cache_file)
	else:
		data = pd.read_csv(csv_file, **kwargs)
		feather.write_dataframe(data, cache_file)
	return data


def main(**opt):
	
	gc.enable()

	# Get the optimized parameters
	n_folds = opt.pop('n_folds', 5)
	tag = opt.pop('tag', '')
	tmt = datetime.now().strftime('%Y%m%d_%H%M')
	tag += '_' + tmt + '_'
	clf_name = opt.get('model', 'GBMClassifier')
	clf = getattr(models, clf_name)(opt)
	assert clf is not None

	# data directory
	cur_dir = op.dirname(__file__)
	data_dir = op.join(cur_dir, '../data')

	# read data
	print("Processing reading raw data ...")
	train_file = op.join(data_dir, 'application_train.csv.zip')
	train = cache_read(train_file)
	test_file = op.join(data_dir, 'application_test.csv.zip')
	test = cache_read(test_file)

	# bu tag
	# bureau_file = op.join(data_dir, 'bureau.csv.zip')
	# bureau = cache_read(bureau_file)
 
	# # bub tag
	# bureau_balance_file = op.join(data_dir, 'bureau_balance.csv.zip')
	# bureau_balance = cache_read(bureau_balance_file)
 
	# # ccb tag
	# credit_card_balance_file = op.join(data_dir, 'credit_card_balance.csv.zip')
	# credit_card_balance = cache_read(credit_card_balance_file)
 
	# # itp tag
	# installments_payments_file = op.join(data_dir, 'installments_payments.csv.zip')
	# installments_payments = cache_read(installments_payments_file)
 
	# # pcb tag
	# pos_cash_balance_file = op.join(data_dir, 'POS_CASH_balance.csv.zip')
	# pos_cash_balance = cache_read(pos_cash_balance_file)
 
	# # pa tag
	# previous_application_file = op.join(data_dir, 'previous_application.csv.zip')
	# previous_app = cache_read(previous_application_file)

	# merge data and make features
	
	# part 1. Concate the train and test data to make features together.
	print("Feature not added train data shape: {}".format(train.shape))
	y = train['TARGET']
	train.drop(['TARGET'], axis=1, inplace=True)
	train_size = train.shape[0]
	train = pd.concat([train, test])
	subm = test[['SK_ID_CURR']]
	del test

	# part 2. add bureau info.
	# part 3. add credit_card_balance, pos_cash_balance, and installments_payments info.
	# part 4. add previous application info. Most only contains 1 application.
	
	# part 5. remove useless feat and fill missing value.
	train = train.fillna(-999)

	# category the variable
	for _f in train.columns.tolist():
		if train[_f].dtype == "object":
			train[_f] = train[_f].astype('category')

	# split the data into train and test
	test = train.iloc[train_size:, :].reset_index(drop=True)
	train = train.iloc[:train_size, :].reset_index(drop=True)
	print("Feature added train shape: {}".format(train.shape))

	# may do some tweak using feature importance
	feat_selected = train.columns.tolist()[1:]

	# do stacking.
	print("Begin to do cross validation to model data ...")
	cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=521)
	train_pred = np.zeros(train.shape[0])
	test_pred = np.zeros((test.shape[0], n_folds))
	feat_imp = pd.DataFrame(np.zeros((len(feat_selected), n_folds)))
	feat_imp['features'] = feat_selected

	for k, (trn_idx, val_idx) in enumerate(cv.split(train, y)):
		trn_x, trn_y = train[feat_selected].iloc[trn_idx], y.iloc[trn_idx]
		val_x, val_y = train[feat_selected].iloc[val_idx], y.iloc[val_idx]

		clf.fit(train_set=(trn_x, trn_y), valid_set=(val_x, val_y))

		train_pred[val_idx] = clf.predict_proba(val_x)
		test_pred[:, k] = clf.predict_proba(test[feat_selected])

		stat = roc_auc_score(val_y, train_pred[val_idx])
		print("K={}, AUC: {:.3f}".format(k+1, stat))

		# collect importance info
		feat_imp.iloc[:, k] = clf.get_feat_imp()

	total_auc = roc_auc_score(y, train_pred)
	print("CV-{} had been done! Total train auc is: {}".format(n_folds, total_auc))
	
	feat_imp['imp_mean'] = feat_imp.iloc[:, :n_folds].mean(axis=1)
	feat_imp['imp_std'] = feat_imp.iloc[:, :n_folds].std(axis=1)
	feat_imp = feat_imp.iloc[:, n_folds:].sort_values('imp_mean', ascending=False)

	# save to files
	print("Begin to save statistic into files")
	stat_dir = op.join(cur_dir, '../stat')
	feat_imp_file = op.join(stat_dir, tag + 'feat_imp.csv')
	feat_imp.to_csv(feat_imp_file, index=False)

	train_pred_ = train[['SK_ID_CURR']]
	train_pred_['TARGET_PRED'] = train_pred
	train_pred_file = op.join(stat_dir, tag + 'train_cv_pred.csv')
	train_pred_.to_csv(train_pred_file, index=False)

	print("Saving test prediction to files ...")
	subm['TARGET'] = np.mean(test_pred, axis=1)
	subm_file = op.join(cur_dir, '../sub', tag + 'subm.csv.gz')
	subm.to_csv(subm_file, index=False, compression='gzip')

	print("All prediction done!")

if __name__ == '__main__':
	fire.Fire(main)