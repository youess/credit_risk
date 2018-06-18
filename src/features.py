# -*- coding: utf-8 -*-


import numpy as np
import functools
from utils import *


def merge_feat_by_customer(a, b):
    return pd.merge(a, b, how='left', on=['SK_ID_CURR'])


def log1p_by_column(df, c):
    """log1p某列并替代"""
    df[c] = np.log1p(df[c])
    df.rename(columns={c: c + '_log1p'}, inplace=True)
    return df


def add_prefix_elem(prefix, l):
    return [prefix + '_' + str(x) for x in l]


def create_basic_feat(data_dir, useless_feat):
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
    y = train['TARGET']
    train.drop(['TARGET'], axis=1, inplace=True)
    train_size = train.shape[0]
    train = pd.concat([train, test])
    del test

    # Now transform current basic features data

    # 合约类型
    c = 'NAME_CONTRACT_TYPE'
    train = dummy_replace(train, c)

    # 性别
    c = 'CODE_GENDER'
    ind = train[c] == 'XNA'
    train.loc[ind, c] = "F"
    train = dummy_replace(train, c, drop_first=True)

    # 是否拥有汽车与房产
    for c in ['FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        train = dummy_replace(train, c, drop_first=True)

    # 孩子数量
    # c = 'CNT_CHILDREN'

    # 客户薪水与贷款额度, 年金贷款额度
    for c in ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']:
        train = log1p_by_column(train, c)

    # 陪伴人类型, 收入类型，教育程度，家庭状态，住房状态
    for c in ['NAME_TYPE_SUITE', 'NAME_INCOME_TYPE',
              'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE']:
        train = dummy_replace(train, c)

    c = 'NAME_TYPE_SUITE'
    selected_cols = add_prefix_elem(c, ['Family', 'Spouse, partner', 'Children'])
    train[c + '_family_board'] = train[selected_cols].sum(axis=1)

    # 出生年龄
    c = 'DAYS_BIRTH'
    age = train[c] / 365
    train['age'] = pd.cut(-age, [19, 25, 40, 60, 70],
                          labels=['young_adult', 'adult', 'middle_aged', 'senior_aged'])
    train = dummy_replace(train, 'age')

    # 工龄
    c = 'DAYS_EMPLOYED'
    work_age = train[c] / 365
    train['work_age'] = pd.cut(-work_age, [-np.inf, 0, 1, 3, 5, 10, 100],
                               labels=['Unknown', 'y1_minus', 'y3_minus', 'y5_minus', 'y10_minus', 'y_inf_minus'])
    train = dummy_replace(train, 'work_age')

    # 职业
    c = 'OCCUPATION_TYPE'
    train[c] = train[c].fillna("UNKNOWN")
    train = dummy_replace(train, c)

    for c in ['WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START']:
        train = dummy_replace(train, c)

    # 单位类型
    c = 'ORGANIZATION_TYPE'
    train = dummy_replace(train, c)

    for c in ['DAYS_LAST_PHONE_CHANGE', 'CNT_FAM_MEMBERS']:
        train[c] = train[c].fillna(int(train[c].median()))

    exclude_column_df(train, useless_feat)

    return train, y, train_size


def create_bureau_feat(data_dir, useless_feat):
    """
    创建征信局数据的特征
    :param data_dir: str
    :param useless_feat: set or list
    :return: pd.DataFrame
    """
    # bu tag，征信局记录
    bureau_file = op.join(data_dir, 'bureau.csv.zip')
    bureau = cache_read(bureau_file)

    # bub tag
    bureau_balance_file = op.join(data_dir, 'bureau_balance.csv.zip')
    bureau_balance = cache_read(bureau_balance_file)

    #
    c = 'MONTHS_BALANCE'
    df = get_group_stat(bureau_balance, ['SK_ID_BUREAU'], c, 'count')
    df_tmp = get_group_stat(bureau_balance, ['SK_ID_BUREAU'], c, 'mean',
                            new_dtype='float32')
    df = df.merge(df_tmp, how='left', on=['SK_ID_BUREAU'])

    c = 'STATUS'
    df_status = get_group_stat(bureau_balance, ['SK_ID_BUREAU', c], 'MONTHS_BALANCE', 'count')
    df_status.rename(columns={df_status.columns.tolist()[-1]: 's'}, inplace=True)
    df_status = pivot_stat_single(df_status, index='SK_ID_BUREAU', column=c, fill_value=0)
    df_status['s1plus'] = df_status[['s_1', 's_2', 's_3', 's_4', 's_5']].sum(axis=1)
    df_status['s2plus'] = df_status[['s_2', 's_3', 's_4', 's_5']].sum(axis=1)

    df = df.merge(df_status, on=['SK_ID_BUREAU'], how='left')
    del df_status

    bureau = bureau.merge(df, on=['SK_ID_BUREAU'], how='left')

    # added_cols_len = len(df.columns.tolist()) - 1
    added_cols = df.columns.tolist()[1:]

    del bureau_balance

    features = []

    # CREDIT PROFILE
    # 每个客户过去贷款次数
    df = get_group_stat(bureau, ['SK_ID_CURR'], 'SK_ID_BUREAU', 'count')
    features.append(df)

    # 活跃贷款百分比
    df_active = get_group_stat(
        bureau[bureau['CREDIT_ACTIVE'] == 'Active'],
        ['SK_ID_CURR'], 'SK_ID_BUREAU', 'count', new_column='active_bureau_id_count')
    c1, c2 = df.columns.tolist()[-1], df_active.columns.tolist()[-1]
    df = merge_feat_by_customer(df_active, df)
    df['ACTIVE_CREDIT_SK_ID_BUREAU_PER'] = (df[c2] / df[c1]).fillna(0)
    df.drop([c1, c2], axis=1, inplace=True)
    del df_active
    features.append(df)

    # 客户信用类型申请次数; 货币种类, 应该关系不大
    for c in ['CREDIT_ACTIVE', 'CREDIT_CURRENCY']:
        df = get_group_stat(bureau, ['SK_ID_CURR', c], 'SK_ID_BUREAU', 'count')
        df = pivot_stat_single(df, index='SK_ID_CURR', column=c, fill_value=0)
        features.append(df)

    # 客户申请天数; 逾期时间
    for c in ['DAYS_CREDIT', 'CREDIT_DAY_OVERDUE']:
        for fun in ['mean', 'median', 'std']:
            df = get_group_stat(bureau, ['SK_ID_CURR'], c, fun, new_dtype='float32')
            features.append(df)

    # 没有结束的申请的个数
    c = 'DAYS_CREDIT_ENDDATE'
    df = get_group_stat(bureau[bureau[c] >= 0], ['SK_ID_CURR'], c, 'count', new_column='NOT_FINISHED_CREDIT_COUNT')
    features.append(df)
    for fun in ['mean', 'median', 'std']:
        df = get_group_stat(bureau[bureau[c] >= 0], ['SK_ID_CURR'], c, fun,
                            new_column='NOT_FINISHED_CREDIT_' + fun,
                            new_dtype='float32')
        features.append(df)

        df = get_group_stat(bureau, ['SK_ID_CURR'], c, fun, new_dtype='float32')
        features.append(df)

    # 对于已经关闭的申请
    c = 'DAYS_ENDDATE_FACT'
    for fun in ['mean', 'median', 'std']:
        df = get_group_stat(bureau, ['SK_ID_CURR'], c, fun, new_dtype='float32')
        features.append(df)

    # 欠款最大数量
    c = 'AMT_CREDIT_MAX_OVERDUE'
    for fun in ['max', 'mean', 'median', 'std']:
        df = get_group_stat(bureau, ['SK_ID_CURR'], c, fun, new_dtype='float32')
        df = log1p_by_column(df, df.columns.tolist()[-1])
        features.append(df)

    # 欠款延长次数
    c = 'CNT_CREDIT_PROLONG'
    for fun in ['sum', 'max', 'mean', 'median', 'std']:
        df = get_group_stat(bureau, ['SK_ID_CURR'], c, fun, new_dtype='float32')
        features.append(df)

    df = get_group_stat(bureau[bureau[c] > 0], ['SK_ID_CURR'], c, 'count',
                        new_column='POS_PROLONG_COUNT')
    features.append(df)

    # 信用额度
    c = 'AMT_CREDIT_SUM'
    for fun in ['max', 'mean', 'median', 'std']:
        df = get_group_stat(bureau, ['SK_ID_CURR'], c, fun, new_dtype='float32')
        df = log1p_by_column(df, df.columns.tolist()[-1])
        features.append(df)

    # 最近一次的信用额度是多少
    df = bureau.groupby(['SK_ID_CURR'])['DAYS_CREDIT'].transform('max')
    flag_latest_id = bureau['DAYS_CREDIT'] == df
    df = get_group_stat(bureau.loc[flag_latest_id, :], ['SK_ID_CURR'], c, 'max',
                        new_column='latest_id_amt_credit_sum',
                        new_dtype='float32')
    df = log1p_by_column(df, df.columns.tolist()[-1])
    features.append(df)

    # 当前贷款数额, 最大信用额度
    for c in ['AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_LIMIT']:
        # < 0的个数
        df = get_group_stat(bureau[bureau[c] < 0], ['SK_ID_CURR'], c, 'count',
                            new_column='neg_' + c + '_count')
        features.append(df)

        for fun in ['mean', 'median']:

            df = get_group_stat(bureau[bureau[c] < 0], ['SK_ID_CURR'], c, fun,
                                new_column='neg_' + c + '_' + fun,
                                new_dtype='float32')
            cc = df.columns.tolist()[-1]
            df[cc] = -df[cc]
            df = log1p_by_column(df, cc)
            cc = df.columns.tolist()[-1]
            df[cc] = -df[cc]
            features.append(df)

            df = get_group_stat(bureau[bureau[c] >= 0], ['SK_ID_CURR'], c, fun,
                                new_column='pos_' + c + '_' + fun,
                                new_dtype='float32')
            df = log1p_by_column(df, df.columns.tolist()[-1])
            features.append(df)

    # 逾期贷款数额, 年金
    for c in ['AMT_CREDIT_SUM_OVERDUE', 'AMT_ANNUITY']:
        for fun in ['mean', 'median', 'std']:
            df = get_group_stat(bureau, ['SK_ID_CURR'], c, fun,
                                new_dtype='float32')
            df = log1p_by_column(df, df.columns.tolist()[-1])
            features.append(df)

    # 每个客户贷款类型数目
    c = 'CREDIT_TYPE'
    df = get_group_stat(bureau, ['SK_ID_CURR'], c, 'nunique')
    features.append(df)

    # 每个客户贷款类型的平均数目
    df = get_group_stat(bureau, ['SK_ID_CURR', c], 'SK_ID_BUREAU', 'count')
    for fun in ['mean', 'median']:
        features.append(
            get_group_stat(df, ['SK_ID_CURR'], df.columns.tolist()[-1], fun,
                           new_column='_'.join(['SK_ID_CURR', c, 'SK_ID_BUREAU', fun]),
                           new_dtype='float32')
        )

    # 天数
    for c in ['DAYS_CREDIT_UPDATE'] + added_cols[:2]:
        for fun in ['max', 'mean', 'median', 'std']:
            df = get_group_stat(bureau, ['SK_ID_CURR'], c, fun,
                                new_dtype='float32')
            features.append(df)

    # 后续添加的列
    # cols = bureau.columns.tolist()[-added_cols_len:]
    for c in added_cols[2:]:
        df = get_group_stat(bureau[~bureau[c].isnull()], ['SK_ID_CURR'], c, 'sum')
        features.append(df)

    # 合在一起
    features = functools.reduce(merge_feat_by_customer, features)

    # 对列名添加标签
    features = add_columns_tag(features, 'bu_', keep=['SK_ID_CURR'])
    exclude_column_df(features, useless_feat)

    del bureau

    return features


def create_credit_card_feat(data_dir, useless_feat):
    """
    信用卡交易信息特征
    :param data_dir:
    :param useless_feat:
    :return:
    """
    credit_card_balance_file = op.join(data_dir, 'credit_card_balance.csv.zip')
    ccb = cache_read(credit_card_balance_file)

    features = []

    # 客户申请记录条数
    df = get_group_stat(ccb, ['SK_ID_CURR'], 'SK_ID_PREV', 'nunique')
    features.append(df)

    # month count
    c = 'MONTHS_BALANCE'
    df = get_group_stat(ccb, ['SK_ID_CURR'], c, 'count')
    features.append(df)

    # money

    c = 'AMT_CREDIT_LIMIT_ACTUAL'
    for fun in ['min', 'max', 'mean', 'median', 'std']:
        df = get_group_stat(ccb, ['SK_ID_CURR'], c, fun, new_dtype='float32')
        cc = df.columns.tolist()[-1]
        df = log1p_by_column(df, cc)
        features.append(df)

    # c = 'AMT_DRAWINGS_ATM_CURRENT'
    # (ccb[c] < 0).sum()    # 1

    # c = 'AMT_DRAWINGS_CURRENT'
    # (ccb[c] < 0).sum()      # 3
    for c in ['AMT_DRAWINGS_ATM_CURRENT', 'AMT_DRAWINGS_CURRENT',
              'AMT_DRAWINGS_OTHER_CURRENT',
              'AMT_DRAWINGS_POS_CURRENT',
              'AMT_INST_MIN_REGULARITY',
              'AMT_PAYMENT_CURRENT',
              'AMT_PAYMENT_TOTAL_CURRENT'
              ]:
        ccb[c] = ccb[c].clip(0, np.inf)
        for fun in ['mean', 'median', 'std']:
            df = get_group_stat(ccb, ['SK_ID_CURR'], c, fun, new_dtype='float32')
            cc = df.columns.tolist()[-1]
            df = log1p_by_column(df, cc)
            features.append(df)

    # treat the negative numbers.
    for c in [
        'AMT_BALANCE',                # 2345
        'AMT_RECEIVABLE_PRINCIPAL',   # 2428
        'AMT_RECIVABLE',              # 109338
        'AMT_TOTAL_RECEIVABLE',       # 109330
    ]:
        cc = 'sign_' + c
        ccb[cc] = np.sign(ccb[c])
        df = get_group_stat(ccb[ccb[cc] < 0], ['SK_ID_CURR'], 'SK_ID_PREV', 'nunique',
                            new_column='neg_nunique_' + cc)
        features.append(df)

        ccb[c] = ccb[c].clip(0, np.inf)
        for fun in ['mean', 'median', 'std']:
            df = get_group_stat(ccb, ['SK_ID_CURR'], c, fun, new_dtype='float32')
            cc = df.columns.tolist()[-1]
            df = log1p_by_column(df, cc)
            features.append(df)

    # mean, median, std
    for c in ['MONTHS_BALANCE', 'CNT_DRAWINGS_ATM_CURRENT', 'CNT_DRAWINGS_CURRENT',
              'CNT_DRAWINGS_OTHER_CURRENT', 'CNT_DRAWINGS_POS_CURRENT', 'CNT_INSTALMENT_MATURE_CUM',
              'SK_DPD', 'SK_DPD_DEF']:
        for fun in ['mean', 'median', 'std']:
            df = get_group_stat(ccb, ['SK_ID_CURR'], c, fun, new_dtype='float32')
            features.append(df)

    for c in ['SK_DPD', 'SK_DPD_DEF']:
        df = get_group_stat(ccb, ['SK_ID_CURR'], c, 'max')
        features.append(df)

    # merge all the features
    features = functools.reduce(merge_feat_by_customer, features)

    # 对列名添加标签
    features = add_columns_tag(features, 'ccb_', keep=['SK_ID_CURR'])
    exclude_column_df(features, useless_feat)
    del ccb

    return features


def create_features(data_dir, useless_feat):
    """
    汇聚所有特征，并返回合适的训练数据
    :param data_dir:
    :param useless_feat:
    :return:
    """

    train_x, y, train_size = create_basic_feat(data_dir, useless_feat)
    bureau_feat = create_bureau_feat(data_dir, useless_feat)
    ccb_feat = create_credit_card_feat(data_dir, useless_feat)

    return bureau_feat


'''
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

    """
    def concate_str(x):
        x = list(set(x))
        x = sorted(x)
        return ','.join(x)

    df_tmp = get_group_stat(
        previous_app, ['SK_ID_CURR'], 'NAME_CONTRACT_TYPE',
        concate_str, new_dtype='object')
    df = df.merge(df_tmp, how='left', on=['SK_ID_CURR'])
    """

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
'''