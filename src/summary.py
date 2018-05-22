# -*- coding: utf-8 -*-
# @Author: denglei
# @Date:   2018-05-20 16:55:24
# @Last Modified by:   denglei
# @Last Modified time: 2018-05-20 17:00:32


import os
# import numpy as np
import glob
import pandas as pd


def print_data_summary(df, name="data"):
    print(f"[{name}] Summary info shape: {df.shape}")
    for col in df.columns.tolist():
        print(f"--{col}: ")
        na_counts = df[col].isnull().sum()
        print(f"--NA count[ {na_counts}, {100*na_counts/df.shape[0]:.2f}% ]",
              end=";")
        if df[col].dtype == object:
            stat = df[col].value_counts()
            print(f"Category count: {list(zip(stat.index, stat.values))}")
        else:
            stat = df[col].quantile([0, 0.25, 0.5, 0.75, 1]).round(4)
            stat = list(zip(stat.index, stat.values))
            stat.append(('nunique', df[col].nunique()))
            stat.append(('mean', df[col].mean().round(4)))
            stat.append(('std', df[col].std().round(4)))
            print(f"Numeric stat: {stat}")
        print("------" * 15)


def main():
    data_dir = os.path.join(os.path.dirname(__file__), '../data')
    # files = os.listdir(data_dir)
    files = glob.glob1(data_dir, '*.csv.zip')
    files = [f for f in files if 'application_test' not in f]
    files = [f for f in files if 'sample_submission' not in f]
    files = [os.path.join(data_dir, f) for f in files]

    for f in files:
        df = pd.read_csv(f)
        print_data_summary(df, name=f)


if __name__ == '__main__':
    main()
