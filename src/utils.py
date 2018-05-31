# -*- coding: utf-8 -*-
# @Author: denis
# @Date:   2018-05-25 11:14:36
# @Last Modified by:   denis
# @Last Modified time: 2018-05-25 11:15:20


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


NA_NEG_VALUE = -9999999
NA_POS_VALUE = 365243
NA_STRING    = 'XNA'


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


def dummy_replace(data, c, **kwargs):
    df = pd.get_dummies(data[c], prefix=c, **kwargs)
    data.drop(c, axis=1, inplace=True)
    data = pd.concat([data, df], axis=1)
    return data


def add_prefix(prefix, list, sep='_'):
    return [prefix + sep + e for e in list]


def fill_na(df, c):
    if df[c].dtype == 'object':
        df[c].fillna(NA_STRING, inplace=True)
    else:
        if 'DAYS' in c:
            df[c].fillna(NA_POS_VALUE, inplace=True)
        else:
            df[c].fillna(NA_NEG_VALUE, inplace=True)