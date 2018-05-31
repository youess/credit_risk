# -*- coding: utf-8 -*-
# @Author: denglei
# @Date:   2018-05-21 09:39:03
# @Last Modified by:   denis
# @Last Modified time: 2018-05-25 14:42:37


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
from utils import *


def create_cur_app_feat(data_dir, useless_feat):
    """
    读取训练和测试集，填补缺失值并进行One-hot encoding特征

    返回train := [train, test] X特征，以及y target，以及原始train数据集的大小
    """
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
    y = train['TARGET']
    train.drop(['TARGET'], axis=1, inplace=True)
    train_size = train.shape[0]
    train = pd.concat([train, test])
    # subm = test[['SK_ID_CURR']]
    del test

    print("Raw train data features shape: {}".format(train.shape))
    
    # fill na with mean value
    cols = ['AMT_ANNUITY', 'AMT_GOODS_PRICE']
    for c in cols:
        train[c].fillna(train[c].mean(), inplace=True)

    c = 'CNT_FAM_MEMBERS'
    train[c].fillna(int(train[c].mean()), inplace=True)
    train[c] = train[c].clip(0, 6)

    # Get NA value impute work done!
    c = 'CODE_GENDER'
    ind = train[c] == 'XNA'
    train.loc[ind, c] = np.random.choice(['F', 'M'], 4)

    c = 'HOUR_APPR_PROCESS_START'
    train[c] = train[c].astype('uint8')

    c = 'NAME_FAMILY_STATUS'
    ind = train[c] == 'Unknown'
    train.loc[ind, c] = 'Married'      # assign to most frequent status

    # Fill NA value for this data
    for c in train.columns.tolist():
        fill_na(train, c)

    # do some feature engneering works.
    # train[c].replace({'Other': NA_STRING}, inplace=True)       # other may not equal to NA_STRING
    # too many just dummy it

    # Get dummy features
    cols = ['FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE',
            'ORGANIZATION_TYPE', 'HOUR_APPR_PROCESS_START',
            'WEEKDAY_APPR_PROCESS_START', 'OCCUPATION_TYPE',
            'NAME_CONTRACT_TYPE', 'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE',
            'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
            'NAME_HOUSING_TYPE']
    for c in cols:
        train = dummy_replace(train, c)

    # for binary category
    cols = ['EMERGENCYSTATE_MODE', 'CODE_GENDER', 
        'FLAG_OWN_REALTY', 'FLAG_OWN_REALTY']
    for c in cols:
        train = dummy_replace(train, c, drop_first=True)

    c = 'NAME_TYPE_SUITE'
    sc = add_prefix(c, ['Children', 'Family', 'Spouse, partner'])
    train[c + '_BoardFamily'] = train[sc].sum(axis=1)
    sc = add_prefix(c, ['Other_A', 'Other_B', 'Group of people'])
    train[c + '_Others'] = train[sc].sum(axis=1)

    c = 'NAME_INCOME_TYPE'
    sc = add_prefix(c, ['Unemployed', 'Maternity leave'])
    train[c + '_Unwork'] = train[sc].sum(axis=1)
    sc = add_prefix(c, ['Businessman', 'Student'])
    train[c + '_Potential'] = train[sc].sum(axis=1) 

    c = 'NAME_EDUCATION_TYPE'
    sc = add_prefix(c, ['Academic degree', 'Higher education', 'Incomplete higher'])
    train[c + '_high_degree'] = train[sc].sum(axis=1)

    c = 'NAME_HOUSING_TYPE'
    sc = add_prefix(c, ['Office apartment', 'House / apartment', 'Co-op apartment'])
    train[c + '_Fix_Apartment'] = train[sc].sum(axis=1)
    sc = add_prefix(c, ['Municipal apartment', 'With parents', 'Rented apartment'])
    train[c + '_Unstable_Apartment'] = train[sc].sum(axis=1)

    c = 'OCCUPATION_TYPE'
    sc = add_prefix(c, ['Accountants', 'High skill tech staff', 'Managers',
                        'Core staff', 'HR staff', 'IT staff'])
    train[c + '_Low_Risk'] = train[sc].sum(axis=1)
    sc = add_prefix(c, ['Private service staff', 'Medicine staff',
                        'Secretaries', 'Realty agents'])
    train[c + '_Mid_Risk'] = train[sc].sum(axis=1)
    sc = add_prefix(c, ['Cleaning staff', 'Sales staff', 'Cooking staff',
                        'Laborers', 'Security staff', 'Waiters/barmen staff', 'Drivers',
                        'Low-skill Laborers'])
    train[c + '_High_Risk'] = train[sc].sum(axis=1)

    # read useless feature list and remove it from data
    train = train[[f_ for f_ in train.columns.tolist()
                   if f_ not in useless_feat]]

    return train, y, train_size 


def create_bureau_feat(data_dir, useless_feat):
    """
    构建征信局数据的特征，并返回
    """
    # bu tag，征信局记录
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
    # 逾期状态超过2以上的数量[2表示逾期30天以上]
    df2['2plus'] = df2[['2', '3', '4', '5']].sum(axis=1)

    df = pd.merge(df1, df2, how='left', on=['SK_ID_BUREAU'])
    del df1, df2, bureau_balance
    # df.sample(5)
    bureau = pd.merge(bureau, df, how='left', on=['SK_ID_BUREAU'])
    del df

    # bureau数据集当中记录的是各个SK_ID_BUREAU的统计信息
    # 
    # 征信局中记录借贷多少次
    df = get_group_stat(
        bureau, ['SK_ID_CURR'], 'SK_ID_BUREAU', fun='count')

    # 借贷状态，申请个数，申请平均天数delta
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
    bureau['CREDIT_ACTIVE2'] = bureau['CREDIT_ACTIVE'].replace({
        'Sold': 'Bad',
        'Bad debt': 'Bad'
    })

    for fun_name in ['mean', 'median', 'std']:
        # 不包含closed的贷款记录
        df_tmp = get_group_stat(
            bureau[bureau['CREDIT_ACTIVE2'] != 'Closed'],
            ['SK_ID_CURR'], 'DAYS_CREDIT', fun=fun_name, new_dtype='float32')
        df = df.merge(df_tmp, how='left', on=['SK_ID_CURR']).fillna(0)

        # 包含closed的
        df_tmp = get_group_stat(
            bureau, ['SK_ID_CURR'], 'DAYS_CREDIT',
            fun=fun_name, new_dtype='float32')
        df = df.merge(df_tmp, how='left', on=['SK_ID_CURR']).fillna(0)

    # 最大借贷逾期时间
    for fun_name in ['max', 'min', 'mean', 'median', 'std']:
        df_tmp = get_group_stat(
            bureau, ['SK_ID_CURR'], 'CREDIT_DAY_OVERDUE',
            fun=fun_name, new_dtype='float32')
        df = df.merge(df_tmp, how='left', on=['SK_ID_CURR'])

    # 从申请这次借贷之后获得CB信用信息的天数, only for closed credit
    cols = ['DAYS_CREDIT_ENDDATE', 'DAYS_ENDDATE_FACT',
            'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_LIMIT',
            'AMT_CREDIT_SUM_OVERDUE', 'DAYS_CREDIT_UPDATE',
            'AMT_ANNUITY', 'MONTHS_BALANCE']
    for c in cols:
        for fun_name in ['mean', 'median', 'std']:
            df_tmp = get_group_stat(
                bureau[['SK_ID_CURR', c]].dropna(),
                ['SK_ID_CURR'], c, fun=fun_name, new_dtype='float32')
            df = df.merge(df_tmp, how='left', on=['SK_ID_CURR'])     # fill nan after

    # AMT_CREDIT_MAX_OVERDUE
    cols = ['AMT_CREDIT_MAX_OVERDUE', 'CNT_CREDIT_PROLONG']
    for c in cols:
        for fun_name in ['max', 'min', 'mean', 'median', 'std']:

            df_tmp = get_group_stat(
                bureau[['SK_ID_CURR', c]].dropna(),
                ['SK_ID_CURR'], c, fun=fun_name, new_dtype='float32')
            # c = df_tmp.columns.tolist()[-1]
            # df_tmp.iloc[:, 1] = df_tmp.iloc[:, 1].apply(np.log1p)
            # df_tmp[c + '_log1p'] = df_tmp[c].apply(np.log1p)
            # df_tmp.drop(c, axis=1, inplace=True)
            df = df.merge(df_tmp, how='left', on=['SK_ID_CURR'])

    #
    c = 'CREDIT_TYPE'
    df_tmp = get_group_stat(
        bureau, ['SK_ID_CURR', c], 'SK_ID_BUREAU', fun='nunique')
    df_tmp = pivot_stat_single(df_tmp, 'SK_ID_CURR', c, fill_value=0)
    df = df.merge(df_tmp, how='left', on=['SK_ID_CURR'])

    cols = ['0', '1', '2', '3', '4', '5', 'C', 'X', '2plus']
    for c in cols:
        df_tmp = get_group_stat(
            bureau[['SK_ID_CURR', c]].dropna(), ['SK_ID_CURR'], c, fun='sum')
        df = df.merge(df_tmp, how='left', on=['SK_ID_CURR'])
    del df_tmp

    # 对列名添加标签
    df = add_columns_tag(df, 'bu_', keep=['SK_ID_CURR'])
    df = exclude_column_df(df, useless_feat)
    del bureau

    return df 


def create_ccb_feat(data_dir, useless_feat):

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
    cols = ['MONTHS_BALANCE', 'AMT_BALANCE',
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
        for fun_name in ['mean', 'median', 'std']:
            df_tmp = get_group_stat(
                credit_card_balance[['SK_ID_CURR', c]].dropna(),
                ['SK_ID_CURR'], c, fun=fun_name, new_dtype='float32')
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
    del df_tmp1, df_tmp2
    df_tmp = pivot_stat_single(df_tmp, 'SK_ID_CURR', c, fill_value=0)
    df = df.merge(df_tmp, how='left', on=['SK_ID_CURR'])

    # 对列名添加标签
    df = add_columns_tag(df, 'ccb_', keep=['SK_ID_CURR'])
    del credit_card_balance
    # exlude not used columns
    df = exclude_column_df(df, useless_feat)
    return df


def create_itp_feat(data_dir, useless_feat):

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
    for fun_name in ['min', 'max', 'mean', 'median', 'std']:
        df_tmp = get_group_stat(
            installments_payments, ['SK_ID_CURR'],
            c, fun=fun_name, new_dtype='float32')
        df = df.merge(df_tmp, how='left', on=['SK_ID_CURR'])

    # 借款和还款时间，相对天数
    cols = ['DAYS_INSTALMENT', 'DAYS_ENTRY_PAYMENT']
    installments_payments['DAYS_DIFF'] = installments_payments[cols[0]] - installments_payments[cols[1]]
    cols += ['DAYS_DIFF']
    for c in cols:
        for fun_name in ['mean', 'median', 'std']:
            df_tmp = get_group_stat(
                installments_payments[['SK_ID_CURR', c]].dropna(),
                ['SK_ID_CURR'], c, fun=fun_name, new_dtype='float32')
            df = df.merge(df_tmp, how='left', on=['SK_ID_CURR'])

    # 借还款金额
    cols = ['AMT_INSTALMENT', 'AMT_PAYMENT']
    installments_payments['AMT_DIFF'] = installments_payments[cols[0]] - installments_payments[cols[1]]
    cols += ['AMT_DIFF']
    for c in cols:
        for fun_name in ['mean', 'median', 'std']:
            df_tmp = get_group_stat(
                installments_payments[['SK_ID_CURR', c]].dropna(),
                ['SK_ID_CURR'], c, fun=fun_name, new_dtype='float32')
            df = df.merge(df_tmp, how='left', on=['SK_ID_CURR'])

    # 对列名添加标签
    df = add_columns_tag(df, 'itp_', keep=['SK_ID_CURR'])
    del installments_payments
    # exlude not used columns
    df = exclude_column_df(df, useless_feat)
    return df


def create_pcb_feat(data_dir, useless_feat):

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
        for fun_name in ['mean', 'median', 'std']:
            df = add_df_stat_single(
                df, pos_cash_balance[['SK_ID_CURR', c]].dropna(),
                ['SK_ID_CURR'], c, fun=fun_name, new_dtype='float32')

    cols = ['SK_DPD', 'SK_DPD_DEF']
    for c in cols:
        df = add_df_stat_single(
            df, pos_cash_balance,
            ['SK_ID_CURR'], c, fun='max', new_dtype='float32')
        df = add_df_stat_single(
            df, pos_cash_balance,
            ['SK_ID_CURR'], c, fun='min', new_dtype='float32')

    # name contract status
    c = 'NAME_CONTRACT_STATUS'
    df_tmp = get_group_stat(
        pos_cash_balance, ['SK_ID_CURR', c], 
        'SK_ID_PREV', fun='nunique')
    df_tmp = pivot_stat_single(df_tmp, 'SK_ID_CURR', c, fill_value=0)
    df = df.merge(df_tmp, how='left', on=['SK_ID_CURR'])

    # 对列名添加标签
    df = add_columns_tag(df, 'pcb_', keep=['SK_ID_CURR'])
    del pos_cash_balance
    # exlude not used columns
    df = exclude_column_df(df, useless_feat)
    return df


def create_pre_app_feat(data_dir, useless_feat):

    previous_application_file = op.join(
        data_dir, 'previous_application.csv.zip')
    previous_app = cache_read(previous_application_file)

    # Here unique equal to count, but count is more efficient.
    df = get_group_stat(previous_app, ['SK_ID_CURR'], 'SK_ID_PREV', 'count')

    '''
    def concate_str(x):
        x = list(set(x))
        x = sorted(x)
        return ','.join(x)

    df_tmp = get_group_stat(
        previous_app, ['SK_ID_CURR'], 'NAME_CONTRACT_TYPE',
        concate_str, new_dtype='object')
    df = df.merge(df_tmp, how='left', on=['SK_ID_CURR'])
    '''

    cols = ['AMT_ANNUITY', 'AMT_APPLICATION', 'AMT_CREDIT', 'AMT_DOWN_PAYMENT',
            'AMT_GOODS_PRICE', 'HOUR_APPR_PROCESS_START',
            'NFLAG_LAST_APPL_IN_DAY', 'RATE_DOWN_PAYMENT',
            'RATE_INTEREST_PRIMARY', 'RATE_INTEREST_PRIVILEGED',
            'DAYS_DECISION', 'SELLERPLACE_AREA',
            'CNT_PAYMENT', 'DAYS_FIRST_DRAWING',
            'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE',
            'DAYS_TERMINATION', 'NFLAG_INSURED_ON_APPROVAL']
    for c in cols:
        for fun_name in ['mean', 'median', 'std']:
            df_tmp = get_group_stat(
                previous_app[['SK_ID_CURR', c]].dropna(), ['SK_ID_CURR'],
                c, fun=fun_name, new_dtype='float32')
            df = df.merge(df_tmp, how='left', on=['SK_ID_CURR'])

    obj_cols = [f for f in previous_app.columns.tolist()
                if previous_app[f].dtype == 'object']
    cols_xap = ['CODE_REJECT_REASON', 'NAME_CASH_LOAN_PURPOSE']

    for c in obj_cols:
        previous_app[c] = previous_app[c].fillna(NA_STRING)
        if c in cols_xap:
            previous_app[c].replace({'XAP': NA_STRING}, inplace=True)
        df_tmp = get_group_stat(
            previous_app, ['SK_ID_CURR', c],
            'SK_ID_PREV', fun='count')
        df_tmp = pivot_stat_single(df_tmp, 'SK_ID_CURR', c, fill_value=0)
        df = df.merge(df_tmp, how='left', on=['SK_ID_CURR'])

        df_tmp = get_group_stat(
            previous_app, ['SK_ID_CURR'], c, fun='nunique')
        df = df.merge(df_tmp, how='left', on=['SK_ID_CURR'])

    # 对列名添加标签
    df = add_columns_tag(df, 'pa_', keep=['SK_ID_CURR'])
    del previous_app
    # exlude not used columns
    df = exclude_column_df(df, useless_feat)
    return df


def create_features(data_dir, useless_feat):

    train, y, train_size = create_cur_app_feat(data_dir, useless_feat)
    print(f"Part-1 feature shape: {train.shape}")   
    
    #################################
    # part 2. add bureau info. 
    #################################
    print("Part-2 bureau info adding ...")
    df = create_bureau_feat(data_dir, useless_feat)
    
    train = pd.merge(train, df, how='left', on=['SK_ID_CURR'])
    del df
    print(f"Part-2 feature shape: {train.shape}")

    ##############################################
    # part 3. add credit_card_balance,
    ##############################################
    # Home credit公司历史借贷记录
    print("Part-3 credit card balance info adding ...")
    df = create_ccb_feat(data_dir, useless_feat)
    train = pd.merge(train, df, how='left', on=['SK_ID_CURR'])
    del df
    print(f"Part-3 feature shape: {train.shape}")

    ###############################################
    # part 4 - itp tag
    ###############################################
    print("Part-4 installments_payments info adding ...")
    df = create_itp_feat(data_dir, useless_feat)
    train = pd.merge(train, df, how='left', on=['SK_ID_CURR'])
    del df
    print(f"Part-4 feature shape: {train.shape}")

    ###############################################
    # Part 5 - pcb tag
    ###############################################
    print("Part-5 POS_CASH_balance info adding ...")
    df = create_pcb_feat(data_dir, useless_feat)
    train = pd.merge(train, df, how='left', on=['SK_ID_CURR'])
    del df
    print(f"Part-5 feature shape: {train.shape}")

    ###############################################
    # part 6. add previous application info. Most only contains 1 application.
    ###############################################
    # pa tag
    print("Part-6, previous app info adding ...")
    df = create_pre_app_feat(data_dir, useless_feat)
    train = pd.merge(train, df, how='left', on=['SK_ID_CURR'])
    del df
    print(f"Part-6 feature shape: {train.shape}")

    # Fill missing value.
    for f_ in train.columns.tolist():
        fill_na(train, f_)

    # category the variable after filled na value
    for f_ in train.columns.tolist():
        if train[f_].dtype == "object":
            train[f_] = train[f_].astype('category')

    # convert whitespace to underline in columns
    train.columns = [f.replace(' ', '_') for f in train.columns.tolist()]

    # split the data into train and test
    train = exclude_column_df(train, useless_feat)
    test = train.iloc[train_size:, :].reset_index(drop=True)
    train = train.iloc[:train_size, :].reset_index(drop=True)
    train = pd.concat([train, y], axis=1)
    # test = pd.concat([test, subm], axis=1)
    feather.write_dataframe(train, op.join(data_dir, 'train_feat_cache.feather'))
    feather.write_dataframe(test, op.join(data_dir, 'test_feat_cache.feather'))
    return train, test    


def main(**opt):

    # 准备工作
    gc.enable()
    np.random.seed(123)

    # data directory
    cur_dir = op.dirname(__file__)
    data_dir = op.join(cur_dir, '../data')

    # Get the optimized parameters
    n_folds = opt.pop('n_folds', 5)
    tag = opt.pop('tag', '')
    tmt = datetime.now().strftime('%Y%m%d_%H%M')
    tag += '_' + tmt + '_'
    useless_feat_file = opt.pop('useless_feat_file', op.join(data_dir, '../stat/dump_feat.txt'))
    clf_name = opt.get('model', 'GBMClassifier')
    tag += clf_name + '_'
    clf = getattr(models, clf_name)(opt)
    assert clf is not None

    # 指定train和test数据缓存文件位置
    train_cache_file = op.join(data_dir, 'train_feat_cache.feather')
    test_cache_file = op.join(data_dir, 'test_feat_cache.feather')

    # load useless feat file that contains features not used to train the model
    useless_feat = load_useless_feat(useless_feat_file)

    if op.exists(train_cache_file) and op.exists(test_cache_file):
        print("Loading train and test feathers cache file ...")
        train = feather.read_dataframe(train_cache_file)
        test  = feather.read_dataframe(test_cache_file)
    else:
        train, test = create_features(data_dir, useless_feat)

    train, y = train.iloc[:, :-1], train['TARGET']
    subm = test[['SK_ID_CURR']]
    print("Feature added train shape: {}".format(train.shape))
    train = exclude_column_df(train, useless_feat)
    test  = exclude_column_df(test, useless_feat)

    # if clf_name in ['RFClassifier', 'ETClassifier', 'XGB_Classifier']:
    #     print("One hot encoding variables ...")
    #     train_size = train.shape[0]
    #     data = pd.concat([train, test])
    #     del train, test

    #     obj_cols = [c for c in data.columns.tolist()[1:] if data[c].dtype == 'object' or data[c].dtype.name == 'category']
    #     # print(obj_cols)
    #     not_obj_cols = [c for c in data.columns.tolist() if c not in obj_cols]
    #     if len(obj_cols) > 0:
    #         one_hot_data = pd.get_dummies(data[obj_cols])
    #         # print(one_hot_data.shape, type(one_hot_data))
    #         data = pd.concat([data[not_obj_cols], one_hot_data], axis=1)
    #         data = exclude_column_df(data, useless_feat)
    #     test = data.iloc[train_size:, :].reset_index(drop=True)
    #     train = data.iloc[:train_size, :].reset_index(drop=True)
    #     del data
    #     print("Encoding done!")
    # elif clf_name == "CBClassifier":
    #     print("Re-initialize the models ... ")
    #     cat_features = [idx for idx, c in enumerate(train.columns.tolist()[1:]) if train[c].dtype.name == 'category']
    #     clf = getattr(models, clf_name)(opt, cat_features)
    #     assert clf is not None

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
