# -*- coding: utf-8 -*-
# @Author: denglei
# @Date:   2018-05-21 09:39:03
# @Last Modified by:   denis
# @Last Modified time: 2018-05-23 09:26:48


# import os
import gc
import os.path as op
import numpy as np
import feather
import pandas as pd
import fire
import time
import models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from datetime import datetime
from contextlib import contextmanager


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


def load_useless_feat(filepath):
    """
    加载后续通过模型筛选出的无用特征
    """
    unused_feat = set()
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line == "" or line.startswith("#"):
                continue
            unused_feat.add(line)
    return unused_feat


def add_columns_tag(df, prefix, keep=['SK_ID_CURR']):

    cols = df.columns.tolist()
    cols = [f_ if f_ in keep else prefix + f_ for f_ in cols]
    df.columns = cols
    return df


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print('[{}] done in {:.2f} s'.format(name, time.time() - t0))


def add_df_column(df, grp, select, fun=None, new_column=None,
                  new_dtype='uint16', reverse_order=False):
    """
    df, a dataframe object,
    grp, a list
    select, a column name from df
    """
    if fun is None:
        fun = 'count'

    if new_column is None:
        new_column = str(fun) + '_' + '_'.join(grp + [select])

    with timer("Counting {}".format(new_column)):
        if fun == 'cumcount':
            if reverse_order:
                df[new_column] = df[
                    grp + [select]].iloc[::-1, :].groupby(
                        grp)[select].agg(fun).astype(new_dtype)
            else:
                df[new_column] = df[grp + [select]].groupby(
                    grp)[select].agg(fun).astype(new_dtype)
        else:
            df = df.merge(
                df[grp + [select]].groupby(
                    grp)[select].agg(fun).astype(
                        new_dtype).reset_index().rename(
                            columns={select: new_column}), how='left')
    return df


def get_group_stat(df, grp, select, fun=None,
                   new_column=None, new_dtype='uint16'):

    if fun is None:
        fun = 'count'

    if new_column is None:
        if not isinstance(fun, str):
            fun_name = fun.__name__
        else:
            fun_name = fun
        new_column = fun_name + '_' + '_'.join(grp + [select])

    # not support cumcout
    with timer("Stating {}".format(new_column)):
        stat_df = df[grp + [select]].groupby(
            grp)[select].agg(fun).astype(new_dtype).\
            reset_index().rename(columns={select: new_column})
        return stat_df


def pivot_stat_single(df, index, column, fill_value=None):
    """
    将两列的统计量reshape成一列并将列名更改
    """
    df = df.pivot_table(index=index, columns=column, fill_value=fill_value)
    cols = []
    for idx0, idx1 in zip(df.columns.labels[0], df.columns.labels[1]):
        col = [df.columns.levels[0][idx0],
               df.columns.levels[1][idx1].replace(' ', '_')]
        cols.append('_'.join(col))

    df.columns = cols
    df = df.reset_index()
    return df


def exclude_column_df(df, exclude_set):
    cols = [f for f in df.columns if f not in exclude_set]
    return df[cols]


def main(**opt):

    # 准备工作
    gc.enable()
    np.random.seed(123)

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

    # merge data and make features
    ######################################################################
    # part 1. Concate the train and test data to make features together.
    ######################################################################
    print("Part-1 Feature not added train data shape: {}".format(train.shape))
    y = train['TARGET']
    train.drop(['TARGET'], axis=1, inplace=True)
    train_size = train.shape[0]
    train = pd.concat([train, test])
    subm = test[['SK_ID_CURR']]
    del test

    # read useless feature list and remove it from data
    useless_feat_file = op.join(cur_dir, '../stat/dump_feat.txt')
    useless_feat = load_useless_feat(useless_feat_file)

    train = train[[f_ for f_ in train.columns.tolist()
                   if f_ not in useless_feat]]

    #################################
    # part 2. add bureau info.
    #################################
    print("Part-2 bureau info adding ...")
    # bu tag
    bureau_file = op.join(data_dir, 'bureau.csv.zip')
    bureau = cache_read(bureau_file)

    # bub tag
    bureau_balance_file = op.join(data_dir, 'bureau_balance.csv.zip')
    bureau_balance = cache_read(bureau_balance_file)

    # construct features
    df1 = bureau_balance.groupby(['SK_ID_BUREAU'])['MONTHS_BALANCE'].count(). \
        astype('uint16').sort_values().reset_index()
    df2 = bureau_balance.groupby(['SK_ID_BUREAU', 'STATUS'])['STATUS']. \
        count().astype('uint16')
    df2 = df2.unstack('STATUS').fillna(0).astype('uint16')
    del df2.columns.name
    df2 = df2.reset_index()
    df = pd.merge(df1, df2, how='left', on=['SK_ID_BUREAU'])
    del df1, df2, bureau_balance
    # df.sample(5)
    bureau = pd.merge(bureau, df, how='left', on=['SK_ID_BUREAU'])
    del df

    # 借贷多少次
    df = get_group_stat(
        bureau, ['SK_ID_CURR'], 'SK_ID_BUREAU',
        fun='nunique', new_dtype='uint16')
    df_tmp = get_group_stat(
        bureau, ['SK_ID_CURR'], 'SK_ID_BUREAU', fun='count')
    df = df.merge(df_tmp, how='left', on=['SK_ID_CURR'])
    # 借贷状态
    df_tmp = get_group_stat(
        bureau, ['SK_ID_CURR', 'CREDIT_ACTIVE'],
        'SK_ID_BUREAU', 'nunique')
    df_tmp = pivot_stat_single(
        df_tmp, 'SK_ID_CURR', 'CREDIT_ACTIVE', fill_value=0)
    df = df.merge(df_tmp, how='left', on=['SK_ID_CURR'])
    # 借贷币种
    df_tmp = get_group_stat(
        bureau, ['SK_ID_CURR'], 'CREDIT_CURRENCY', fun='nunique')
    df = df.merge(df_tmp, how='left', on=['SK_ID_CURR'])

    df_tmp = get_group_stat(
        bureau, ['SK_ID_CURR', 'CREDIT_CURRENCY'],
        'SK_ID_BUREAU', fun='nunique')
    df_tmp = pivot_stat_single(
        df_tmp, 'SK_ID_CURR', 'CREDIT_CURRENCY', fill_value=0)
    df = df.merge(df_tmp, how='left', on=['SK_ID_CURR'])
    # 平均借贷时间，除了closed的
    df_tmp = get_group_stat(
        bureau[bureau['CREDIT_ACTIVE'] != 'Closed'],
        ['SK_ID_CURR'], 'DAYS_CREDIT', fun='mean', new_dtype='float32')
    df = df.merge(df_tmp, how='left', on=['SK_ID_CURR']).fillna(0)
    df_tmp = get_group_stat(
        bureau[bureau['CREDIT_ACTIVE'] != 'Closed'],
        ['SK_ID_CURR'], 'DAYS_CREDIT', fun='std', new_dtype='float32')
    df = df.merge(df_tmp, how='left', on=['SK_ID_CURR']).fillna(0)
    # 最大借贷预期时间
    df_tmp = get_group_stat(
        bureau, ['SK_ID_CURR'], 'CREDIT_DAY_OVERDUE', fun='max')
    df = df.merge(df_tmp, how='left', on=['SK_ID_CURR'])
    # 从申请这次借贷之后获得CB信用信息的天数, only for closed credit
    cols = ['DAYS_CREDIT_ENDDATE', 'DAYS_ENDDATE_FACT',
            'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_LIMIT',
            'AMT_CREDIT_SUM_OVERDUE', 'DAYS_CREDIT_UPDATE',
            'AMT_ANNUITY', 'MONTHS_BALANCE']
    for c in cols:
        df_tmp = get_group_stat(
            bureau[['SK_ID_CURR', c]].dropna(),
            ['SK_ID_CURR'], c, fun='mean', new_dtype='float32')
        df = df.merge(
            df_tmp, how='left', on=['SK_ID_CURR'])     # fill nan after
        df_tmp = get_group_stat(
            bureau[['SK_ID_CURR', c]].dropna(),
            ['SK_ID_CURR'], c, fun='std', new_dtype='float32')
        df = df.merge(df_tmp, how='left', on=['SK_ID_CURR'])

    # AMT_CREDIT_MAX_OVERDUE
    c = 'AMT_CREDIT_MAX_OVERDUE'
    df_tmp = get_group_stat(
        bureau[['SK_ID_CURR', c]].dropna(),
        ['SK_ID_CURR'], c, fun='max', new_dtype='float32')
    df_tmp.iloc[:, 1] = df_tmp.iloc[:, 1].apply(np.log1p)
    df = df.merge(df_tmp, how='left', on=['SK_ID_CURR'])

    # CNT_CREDIT_PROLONG
    c = 'CNT_CREDIT_PROLONG'
    df_tmp = get_group_stat(
        bureau[['SK_ID_CURR', c]].dropna(),
        ['SK_ID_CURR'], c, fun='max')
    df = df.merge(df_tmp, how='left', on=['SK_ID_CURR'])

    #
    c = 'CREDIT_TYPE'
    df_tmp = get_group_stat(
        bureau, ['SK_ID_CURR', c], 'SK_ID_BUREAU', fun='nunique')
    df_tmp = pivot_stat_single(df_tmp, 'SK_ID_CURR', c, fill_value=0)
    df = df.merge(df_tmp, how='left', on=['SK_ID_CURR'])

    cols = ['0', '1', '2', '3', '4', '5', 'C', 'X']
    for c in cols:
        df_tmp = get_group_stat(
            bureau[['SK_ID_CURR', c]].dropna(), ['SK_ID_CURR'], c, fun='sum')
        df = df.merge(df_tmp, how='left', on=['SK_ID_CURR'])
    del df_tmp

    # 对列名添加标签
    df = add_columns_tag(df, 'bu_', keep=['SK_ID_CURR'])
    train = pd.merge(train, df, how='left', on=['SK_ID_CURR'])
    del df, bureau
    # exlude not used columns
    train = exclude_column_df(train, useless_feat)

    ##############################################
    # part 3. add credit_card_balance,
    ##############################################
    print("Part-3 credit card balance info adding ...")
    # ccb tag
    credit_card_balance_file = op.join(data_dir, 'credit_card_balance.csv.zip')
    credit_card_balance = cache_read(credit_card_balance_file)

    # 过去申请的贷款次数
    c = 'SK_ID_PREV'
    df = get_group_stat(credit_card_balance, ['SK_ID_CURR'], c, fun='nunique')
    df_tmp = get_group_stat(
        credit_card_balance, ['SK_ID_CURR'], c, fun='count')
    df = df.merge(df_tmp, how='left', on=['SK_ID_CURR'])

    # 其他指标的均值和标准差
    cols = ['MONTHS_BALANCE', 'AMT_BALANCE', 'AMT_CREDIT_LIMIT_ACTUAL',
            'AMT_CREDIT_LIMIT_ACTUAL', 'AMT_DRAWINGS_ATM_CURRENT',
            'AMT_DRAWINGS_CURRENT', 'AMT_DRAWINGS_OTHER_CURRENT',
            'AMT_DRAWINGS_POS_CURRENT', 'AMT_INST_MIN_REGULARITY',
            'AMT_PAYMENT_CURRENT', 'AMT_PAYMENT_TOTAL_CURRENT',
            'AMT_RECEIVABLE_PRINCIPAL', 'AMT_RECIVABLE',
            'AMT_TOTAL_RECEIVABLE',
            'CNT_DRAWINGS_ATM_CURRENT', 'CNT_DRAWINGS_CURRENT',
            'CNT_DRAWINGS_OTHER_CURRENT', 'CNT_DRAWINGS_POS_CURRENT',
            'CNT_INSTALMENT_MATURE_CUM', 'SK_DPD',
            'SK_DPD_DEF']
    for c in cols:
        df_tmp = get_group_stat(
            credit_card_balance[['SK_ID_CURR', c]].dropna(),
            ['SK_ID_CURR'], c, fun='mean', new_dtype='float32')
        df = df.merge(df_tmp, how='left', on=['SK_ID_CURR'])
        df_tmp = get_group_stat(
            credit_card_balance[['SK_ID_CURR', c]].dropna(),
            ['SK_ID_CURR'], c, fun='std', new_dtype='float32')
        df = df.merge(df_tmp, how='left', on=['SK_ID_CURR'])
    del df_tmp

    #
    c = 'NAME_CONTRACT_STATUS'
    df_tmp = get_group_stat(
        credit_card_balance, ['SK_ID_CURR', c], 'SK_ID_PREV', fun='count')
    df_tmp1 = get_group_stat(
        df_tmp, ['SK_ID_CURR'], df_tmp.columns.tolist()[-1], 'max')
    df_tmp2 = get_group_stat(
        df_tmp, ['SK_ID_CURR'], df_tmp.columns.tolist()[-1], 'min')
    df = df.merge(df_tmp1, how='left', on=['SK_ID_CURR'])
    df = df.merge(df_tmp2, how='left', on=['SK_ID_CURR'])
    del df_tmp1, df_tmp2, df_tmp

    # 对列名添加标签
    df = add_columns_tag(df, 'ccb_', keep=['SK_ID_CURR'])
    train = pd.merge(train, df, how='left', on=['SK_ID_CURR'])
    del df, credit_card_balance
    # exlude not used columns
    train = exclude_column_df(train, useless_feat)

    ###############################################
    # part 4 - itp tag
    ###############################################
    print("Part-4 installments_payments info adding ...")
    installments_payments_file = op.join(
        data_dir, 'installments_payments.csv.zip')
    installments_payments = cache_read(installments_payments_file)

    c = 'SK_ID_PREV'
    df = get_group_stat(installments_payments, ['SK_ID_CURR'],
                        c, fun='nunique')
    df_tmp = get_group_stat(installments_payments,
                            ['SK_ID_CURR'], c, fun='count')
    df = df.merge(df_tmp, how='left', on=['SK_ID_CURR'])

    # 还款方式更改次数
    c = 'NUM_INSTALMENT_VERSION'
    df_tmp = get_group_stat(
        installments_payments, ['SK_ID_CURR'],
        c, fun='nunique')
    df = df.merge(df_tmp, how='left', on=['SK_ID_CURR'])
    
    # 最大和最小分期次数
    c = 'NUM_INSTALMENT_NUMBER'
    df_tmp = get_group_stat(
        installments_payments, ['SK_ID_CURR', 'SK_ID_PREV'],
        c, fun='max')
    c = df_tmp.columns.tolist()[-1]
    df_tmp1 = get_group_stat(df_tmp, ['SK_ID_CURR'], c, 'min')
    df = df.merge(df_tmp1, how='left', on=['SK_ID_CURR'])
    df_tmp2 = get_group_stat(df_tmp, ['SK_ID_CURR'], c, 'max')
    df = df.merge(df_tmp2, how='left', on=['SK_ID_CURR'])
    del df_tmp1, df_tmp2, df_tmp

    # 借款和还款时间，相对天数
    cols = ['DAYS_INSTALMENT', 'DAYS_ENTRY_PAYMENT']
    installments_payments['DAYS_DIFF'] = installments_payments[cols[0]] - installments_payments[cols[1]]
    cols += ['DAYS_DIFF']
    for c in cols:
        df_tmp = get_group_stat(
            installments_payments[['SK_ID_CURR', c]].dropna(), 
            ['SK_ID_CURR'], c, fun='mean', new_dtype='float32')
        df = df.merge(df_tmp, how='left', on=['SK_ID_CURR'])
        df_tmp = get_group_stat(
            installments_payments[['SK_ID_CURR', c]].dropna(), 
            ['SK_ID_CURR'], c, fun='std', new_dtype='float32')
        df = df.merge(df_tmp, how='left', on=['SK_ID_CURR'])

    # 借还款金额
    cols = ['AMT_INSTALMENT', 'AMT_PAYMENT']
    installments_payments['AMT_DIFF'] = installments_payments[cols[0]] - installments_payments[cols[1]]
    cols += ['AMT_DIFF']
    for c in cols:
        df_tmp = get_group_stat(
            installments_payments[['SK_ID_CURR', c]].dropna(), 
            ['SK_ID_CURR'], c, fun='mean', new_dtype='float32')
        df = df.merge(df_tmp, how='left', on=['SK_ID_CURR'])
        df_tmp = get_group_stat(
            installments_payments[['SK_ID_CURR', c]].dropna(), 
            ['SK_ID_CURR'], c, fun='std', new_dtype='float32')
        df = df.merge(df_tmp, how='left', on=['SK_ID_CURR'])

    # 对列名添加标签
    df = add_columns_tag(df, 'itp_', keep=['SK_ID_CURR'])
    train = pd.merge(train, df, how='left', on=['SK_ID_CURR'])
    del df, installments_payments
    # exlude not used columns
    train = exclude_column_df(train, useless_feat)

    ###############################################
    # Part 5 - pcb tag
    ###############################################
    print("Part-5 POS_CASH_balance info adding ...")
    pos_cash_balance_file = op.join(data_dir, 'POS_CASH_balance.csv.zip')
    pos_cash_balance = cache_read(pos_cash_balance_file)

    df = pos_cash_balance[['SK_ID_CURR']].drop_duplicates()

    def add_df_stat_single(df, data, grp, sel, fun, new_dtype='uint16'):
        df_tmp = get_group_stat(data, grp, sel, fun=fun, new_dtype=new_dtype)
        df = df.merge(df_tmp, how='left', on=['SK_ID_CURR'])
        return df

    # MONTH balance
    cols = ['MONTHS_BALANCE', 'CNT_INSTALMENT', 
        'CNT_INSTALMENT_FUTURE', 'SK_DPD', 'SK_DPD_DEF']
    for c in cols:
        df = add_df_stat_single(
            df, pos_cash_balance[['SK_ID_CURR', c]].dropna(),
            ['SK_ID_CURR'], c, fun='mean', new_dtype='float32')
        df = add_df_stat_single(
            df, pos_cash_balance[['SK_ID_CURR', c]].dropna(), 
            ['SK_ID_CURR'], c, fun='std', new_dtype='float32')

    cols = ['SK_DPD', 'SK_DPD_DEF']
    for c in cols:
        df = add_df_stat_single(
            df, pos_cash_balance,
            ['SK_ID_CURR'], c, fun='max')

    # name contract status
    c = 'NAME_CONTRACT_STATUS'
    df_tmp = get_group_stat(
        pos_cash_balance, ['SK_ID_CURR', c], 
        'SK_ID_PREV', fun='nunique')
    df_tmp = pivot_stat_single(df_tmp, 'SK_ID_CURR', c, fill_value=0)
    df = df.merge(df_tmp, how='left', on=['SK_ID_CURR'])


    # 对列名添加标签
    df = add_columns_tag(df, 'pcb_', keep=['SK_ID_CURR'])
    train = pd.merge(train, df, how='left', on=['SK_ID_CURR'])
    del df, pos_cash_balance
    # exlude not used columns
    train = exclude_column_df(train, useless_feat)

    ###############################################
    # part 6. add previous application info. Most only contains 1 application.
    ###############################################
    # pa tag
    print("Part-6, previous app info adding ...")
    previous_application_file = op.join(
        data_dir, 'previous_application.csv.zip')
    previous_app = cache_read(previous_application_file)

    df = get_group_stat(previous_app, ['SK_ID_CURR'], 'SK_ID_PREV', 'count')

    def concate_str(x):
        x = list(set(x))
        x = sorted(x)
        return ','.join(x)

    df_tmp = get_group_stat(
        previous_app, ['SK_ID_CURR'], 'NAME_CONTRACT_TYPE',
        concate_str, new_dtype='object')
    df = df.merge(df_tmp, how='left', on=['SK_ID_CURR'])

    cols = ['AMT_ANNUITY', 'AMT_APPLICATION', 'AMT_CREDIT', 'AMT_DOWN_PAYMENT',
            'AMT_GOODS_PRICE', 'HOUR_APPR_PROCESS_START',
            'NFLAG_LAST_APPL_IN_DAY', 'RATE_DOWN_PAYMENT',
            'RATE_INTEREST_PRIMARY', 'RATE_INTEREST_PRIVILEGED',
            'DAYS_DECISION', 'SELLERPLACE_AREA',
            'CNT_PAYMENT', 'DAYS_FIRST_DRAWING',
            'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE',
            'DAYS_TERMINATION', 'NFLAG_INSURED_ON_APPROVAL']
    for c in cols:
        df_tmp = get_group_stat(
            previous_app[['SK_ID_CURR', c]].dropna(), ['SK_ID_CURR'],
            c, fun='mean', new_dtype='float32')
        df = df.merge(df_tmp, how='left', on=['SK_ID_CURR'])
        df_tmp = get_group_stat(
            previous_app[['SK_ID_CURR', c]].dropna(), ['SK_ID_CURR'],
            c, fun='std', new_dtype='float32')
        df = df.merge(df_tmp, how='left', on=['SK_ID_CURR'])  # fill nan after

    obj_cols = [f for f in previous_app.columns.tolist()
                if previous_app[f].dtype == 'object']
    for c in obj_cols:
        previous_app[c] = previous_app[c].fillna('XNA')
        df_tmp = get_group_stat(
            previous_app, ['SK_ID_CURR', c],
            'SK_ID_PREV', fun='nunique')
        df_tmp = pivot_stat_single(df_tmp, 'SK_ID_CURR', c, fill_value=0)
        df = df.merge(df_tmp, how='left', on=['SK_ID_CURR'])

    # 对列名添加标签
    df = add_columns_tag(df, 'pa_', keep=['SK_ID_CURR'])
    train = pd.merge(train, df, how='left', on=['SK_ID_CURR'])
    del df, previous_app
    # exlude not used columns
    train = exclude_column_df(train, useless_feat)

    # part 5. fill missing value.
    for f_ in train.columns.tolist():
        if train[f_].dtype == 'object':
            train[f_] = train[f_].fillna('XNA')
            # train[f_] = train[f_].astype('category')
        else:
            if 'DAYS' in f_:
                train[f_] = train[f_].fillna(365243)   # inidicate infinity
            else:
                train[f_] = train[f_].fillna(-99999)   # denote na value

    # category the variable after filled na value
    for f_ in train.columns.tolist():
        if train[f_].dtype == "object":
            train[f_] = train[f_].astype('category')

    # split the data into train and test
    test = train.iloc[train_size:, :].reset_index(drop=True)
    train = train.iloc[:train_size, :].reset_index(drop=True)
    print("Feature added train shape: {}".format(train.shape))

    # may do some tweak using feature importance
    train = exclude_column_df(train, useless_feat)
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
        print("K={}, AUC: {:.3f}".format(k+1, stat))

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
