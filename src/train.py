# -*- coding: utf-8 -*-
# @Author: denglei
# @Date:   2018-05-21 09:39:03
# @Last Modified by:   denis
# @Last Modified time: 2018-05-25 14:42:37


import gc
import fire
import numpy as np
import models
from datetime import datetime
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from features import create_features
from utils import *


def main(**opt):
    # 准备工作
    gc.enable()
    np.random.seed(123)

    # data directory
    cur_dir = op.dirname(__file__)
    data_dir = op.join(cur_dir, '../data')

    # Get the optimized parameters
    n_folds = opt.pop('n_folds', 3)
    tag = opt.pop('tag', '')
    tmt = datetime.now().strftime('%Y%m%d_%H%M')
    tag += '_' + tmt + '_'
    clf_name = opt.get('model', 'GBMClassifier')
    tag += clf_name + '_'
    clf = getattr(models, clf_name)(opt)
    assert clf is not None

    feat_file_default = op.join(data_dir, '../stat/feature_list.txt')
    useless_feat_file = opt.get("feat_file", feat_file_default)

    # 指定train和test数据缓存文件位置
    train_cache_file = op.join(data_dir, 'train_feat_cache.feather')
    test_cache_file = op.join(data_dir, 'test_feat_cache.feather')

    # 加载特征列表
    useless_feat = load_feat(useless_feat_file)

    if op.exists(train_cache_file) and op.exists(test_cache_file):
        print("Loading train and test feathers cache file ...")
        train = feather.read_dataframe(train_cache_file)
        test = feather.read_dataframe(test_cache_file)
    else:
        train, test = create_features(data_dir, useless_feat)

    train, y = train.iloc[:, :-1], train['TARGET']
    subm = test[['SK_ID_CURR']]
    print("Feature added train shape: {}".format(train.shape))
    exclude_column_df(train, useless_feat)
    exclude_column_df(test, useless_feat)

    # may do some tweak using feature importance
    feat_selected = train.columns.tolist()[1:]
    print("Used features count: {}".format(len(feat_selected)))

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
        print("K={}, AUC: {:.3f}".format(k + 1, stat))

        # collect importance info
        feat_imp.iloc[:, k] = clf.get_feat_imp()

    total_auc = roc_auc_score(y, train_pred)
    print("CV-{} had been done! Total train auc is: {:.4f}".format(
        n_folds, total_auc))

    feat_imp['imp_mean'] = feat_imp.iloc[:, :n_folds].mean(axis=1)
    feat_imp['imp_std'] = feat_imp.iloc[:, :n_folds].std(axis=1)
    feat_imp['imp_cv'] = feat_imp['imp_std'] / feat_imp['imp_mean']
    feat_imp = feat_imp.iloc[:, n_folds:].sort_values(
        'imp_cv', ascending=True, na_position='last')
    ind1 = feat_imp['imp_cv'].isnull()
    ind2 = feat_imp['imp_cv'] > 0.5
    ind3 = feat_imp['imp_mean'] < 10
    ind = ind1 | (ind2 & ind3)
    feat_imp['should_filter'] = ind.astype('int')

    # save to files
    tag += 'kfold_{}_auc_{:.4f}_'.format(n_folds, total_auc)
    print("Begin to save statistic into files")
    stat_dir = op.join(cur_dir, '../stat')
    feat_imp_file = op.join(stat_dir, tag + 'feat_imp.csv')
    feat_imp.to_csv(feat_imp_file, index=False)

    train_pred_ = train[['SK_ID_CURR']]
    train_pred_.loc[:, 'TARGET_PRED'] = train_pred
    train_pred_file = op.join(stat_dir, tag + 'train_cv_pred.csv')
    train_pred_.to_csv(train_pred_file, index=False)

    print("Saving test prediction to files ...")
    subm['TARGET'] = np.mean(test_pred, axis=1)
    subm_file = op.join(cur_dir, '../sub', tag + 'subm.csv.gz')
    subm.to_csv(subm_file, index=False, compression='gzip')

    print("All prediction done!")


if __name__ == '__main__':
    fire.Fire(main)
