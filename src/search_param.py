# -*- coding: utf-8 -*-
# @Author: denis
# @Date:   2018-05-25 11:13:16
# @Last Modified by:   denis
# @Last Modified time: 2018-05-25 16:40:44


import numpy as np
import pandas as pd
import feather
import os.path as op
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import gc
import fire
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from utils import *
from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedKFold


ITERATIONS = 60
N_THREAD = 4


def load_data(data_dir, useless_feat):

    train_cache_file = op.join(data_dir, 'train_feat_cache.feather')
    test_cache_file = op.join(data_dir, 'test_feat_cache.feather')
    train = feather.read_dataframe(train_cache_file)
    test  = feather.read_dataframe(test_cache_file)
    train, y = train.iloc[:, :-1], train['TARGET']
    train = exclude_column_df(train, useless_feat)
    test  = exclude_column_df(test, useless_feat)
    return train, test, y


def search_opt_params(X, y, estimator, spaces):

    def status_print(optim_result):
        """Status callback durring bayesian hyperparameter search"""
    
        # Get all the models tested so far in DataFrame format
        all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)    
    
        # Get current parameters and the best parameters    
        best_params = pd.Series(bayes_cv_tuner.best_params_)
        print('Model #{}\nBest ROC-AUC: {}\nBest params: {}\n'.format(
            len(all_models),
            np.round(bayes_cv_tuner.best_score_, 4),
            bayes_cv_tuner.best_params_
        ))
    
        # Save all model results
        clf_name = bayes_cv_tuner.estimator.__class__.__name__
        all_models.to_csv(clf_name+"_cv_results.csv")

    print("Begin bayesian cv tuning ...")
    bayes_cv_tuner = BayesSearchCV(
        estimator=estimator,
        search_spaces=spaces,
        scoring='roc_auc',
        cv=StratifiedKFold(
            n_splits=3,
            shuffle=True,
            random_state=42),
        n_jobs=1,
        n_iter=ITERATIONS,
        verbose=0,
        refit=True,
        random_state=18520
    )
    res = bayes_cv_tuner.fit(X, y, callback=status_print)
    print("Done bayesian tuning.")
    return res


def search_gbm(train, y):

    search_opt_params(train, y,
        estimator=lgb.LGBMClassifier(
            objective='binary',
            learning_rate=0.07,
            metric='auc',
            n_jobs=N_THREAD,
            verbose=-1),
        spaces={
            # 'learning_rate': (0.01, 0.3, 'log-uniform'),
            'num_leaves': (32, 255),
            'max_depth': (5, 10),
            'max_bin': (100, 500),
            'subsample': (0.5, 1.0, 'uniform'),
            'subsample_freq': (1, 10),
            'colsample_bytree': (0.5, 1.0, 'uniform'),
            'min_child_weight': (0, 10),
            'subsample_for_bin': (100000, 500000),
            'reg_lambda': (1e-9, 1000, 'log-uniform'),
            'reg_alpha': (1e-9, 1, 'log-uniform'),
            'scale_pos_weight': (1, 500, 'log-uniform'),
            'n_estimators': (50, 100),
        })

def search_lr(train, y):

    search_opt_params(train, y,
        estimator=LogisticRegression(
            solver='sag',
            n_jobs=N_THREAD,
            verbose=0),
        spaces={
            'C': (1e-9, 1e6, 'log-uniform')
            })


def search_rf(train, y):

    search_opt_params(train, y,
        estimator=RandomForestClassifier(
            verbose=0, random_state=127127, n_jobs=N_THREAD),
        spaces={
            'n_estimators': (10, 200),
            'max_features': ['auto', 'sqrt', 0.2],
            'min_samples_leaf': (1, 100)
        })


def search_xgb(train, y):

    search_opt_params(train, y,
        estimator=xgb.XGBClassifier(
            n_jobs=N_THREAD,
            objective='binary:logistic',
            eval_metric='auc',
            learning_rate=0.07,
            silent=0,
            tree_method='approx'),
        spaces={
            # 'learning_rate': (0.01, 1.0, 'log-uniform'),
            'min_child_weight': (0, 10),
            'max_depth': (0, 10),
            'max_delta_step': (0, 20),
            'subsample': (0.01, 1.0, 'uniform'),
            'colsample_bytree': (0.01, 1.0, 'uniform'),
            'colsample_bylevel': (0.01, 1.0, 'uniform'),
            'reg_lambda': (1e-9, 1000, 'log-uniform'),
            'reg_alpha': (1e-9, 1.0, 'log-uniform'),
            'gamma': (1e-9, 0.5, 'log-uniform'),
            'min_child_weight': (0, 5),
            'n_estimators': (50, 100),
            'scale_pos_weight': (1e-6, 500, 'log-uniform')
        })


def search_cb(train, y):

    search_opt_params(train, y,
        estimator=cb.CatBoostClassifier(
            bootstrap_type='Bernoulli',
            eval_metric='AUC',
            od_type='Iter',
            od_wait=45,
            learning_rate=0.07,
            random_seed=17,
            verbose=0,
            n_jobs=N_THREAD,
            allow_writing_files=False),
        spaces={
            'iterations': (50, 100),
            # 'learning_rate': (0.01, 1, 'log-uniform'),
            'depth': (5, 10),
            'l2_leaf_reg': (0.01, 50),
            'subsample': (0.5, 0.9),
            'scale_pos_weight': (1, 20)
        })


def main(**opt):

    gc.enable()
    cur_dir = op.dirname(__file__)
    data_dir = op.join(cur_dir, '../data')

    model_selected = opt.get('model', 'gbm')

    # load useless feat file that contains features not used to train the model
    useless_feat_file = op.join(data_dir, '../stat/dump_feat.txt')
    useless_feat = load_useless_feat(useless_feat_file)

    train, test, y = load_data(data_dir, useless_feat)
    del test
    print("Data Loaded!")

    if model_selected == 'gbm':
        search_gbm(train, y)
    elif model_selected == 'rf':
        search_rf(train, y)
    elif model_selected == 'cb':
        search_cb(train, y)
    elif model_selected == 'xgb':
        search_xgb(train, y)
    else:
        print(model_selected)
        raise("Unknown model, please use one of [gbm, rf, cb, xgb]!")

if __name__ == '__main__':
    fire.Fire(main)