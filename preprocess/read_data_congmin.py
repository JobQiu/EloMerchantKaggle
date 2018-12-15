#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 16:53:59 2018

@author: xavier.qiu
"""
import pandas as pd
import datetime
import gc
import numpy as np
import featuretools as ft
import os
from util.util import compress_int, send_msg


class DataSet(object):

    def __init__(self, data_dir='/content/EloMerchantKaggle/data/'):
        self.data_dir = data_dir
        self.train_x_path = os.path.join(self.data_dir, 'x_train_agg')
        self.test_x_path = os.path.join(self.data_dir, 'x_test_agg')
        self.train_y_path = os.path.join(self.data_dir, 'y_train')

        pass

    def get_train_dataset(self, reset=False, load=True):
        if load and os.path.isfile(self.train_x_path) and os.path.isfile(self.train_y_path):
            return pd.read_csv(self.train_x_path), pd.read_csv(self.train_y_path)

        train_df, hist_df_train, new_trans_df_train = split_trans_into_train_test(data_dir=self.data_dir,
                                                                                  reset=reset)

        return agg(train_df, hist_df_train, new_trans_df_train, True, self.train_x_path, self.train_y_path)

    def get_test_dataset(self, load=True):

        if load and os.path.isfile(self.test_x_path):
            return pd.read_csv(self.test_x_path), None

        print("loading test.csv ...")
        d = {'feature_1': np.uint8, 'feature_2': np.uint8, 'feature_3': np.bool_}
        test_df = pd.read_csv(os.path.join(self.data_dir, "test.csv"), parse_dates=["first_active_month"], dtype=d)
        test_df.info(memory_usage='deep')
        hist_df_test = pd.read_csv(os.path.join(self.data_dir, "historical_transactions_test.csv"),
                                   parse_dates=["purchase_date"])
        hist_df_test = compress_int(hist_df_test)
        new_trans_df_test = pd.read_csv(os.path.join(self.data_dir, "new_merchant_transactions_test.csv"),
                                        parse_dates=["purchase_date"])
        new_trans_df_test = compress_int(new_trans_df_test)
        send_msg("load done")

        return agg(test_df, hist_df_test, new_trans_df_test, False, self.test_x_path, None)


def agg(train_df, hist_df, new_trans_df, isTrain, x_save_path, y_save_path):
    train_df = train_df.copy(deep=True)
    if isTrain:
        target = train_df['target']
        del train_df['target']
    else:
        target = None

    es_train = ft.EntitySet(id='es_train')
    es_train = es_train.entity_from_dataframe(entity_id='train', dataframe=train_df,
                                              index='', time_index='first_active_month')
    es_train = es_train.entity_from_dataframe(entity_id='history', dataframe=hist_df,
                                              index='', time_index='purchase_date')
    es_train = es_train.entity_from_dataframe(entity_id='new_trans', dataframe=new_trans_df,
                                              index='', time_index='purchase_date')
    # Relationship between clients and previous loans
    r_client_previous = ft.Relationship(es_train['train']['card_id'],
                                        es_train['history']['card_id'])

    # Add the relationship to the entity set
    es_train = es_train.add_relationship(r_client_previous)
    r_client_previous = ft.Relationship(es_train['train']['card_id'],
                                        es_train['new_trans']['card_id'])

    # Add the relationship to the entity set
    es_train = es_train.add_relationship(r_client_previous)
    print(" dfs ing ... ")
    x_train, _ = ft.dfs(entityset=es_train,
                        target_entity='train',
                        max_depth=2)
    send_msg("dfs done! ")
    print("saving...")
    if target:
        target.to_csv(y_save_path)
        x_train['index'] = target.index
        x_train.set_index('index')
    x_train.to_csv(x_save_path)

    return x_train, target


def split_trans_into_train_test(data_dir='/content/EloMerchantKaggle/data/', reset=False):
    d = {'feature_1': np.uint8, 'feature_2': np.uint8, 'feature_3': np.bool_}
    print("loading train.csv ...")
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"), parse_dates=["first_active_month"], dtype=d)
    train_df.info(memory_usage='deep')

    if not reset and os.path.isfile(os.path.join(data_dir, "historical_transactions_train.csv")) and os.path.isfile(
            os.path.join(data_dir, "new_merchant_transactions_train.csv")):
        hist_df_train = pd.read_csv(os.path.join(data_dir, "historical_transactions_train.csv"),
                                    parse_dates=["purchase_date"])
        hist_df_train = compress_int(hist_df_train)
        new_trans_df_train = pd.read_csv(os.path.join(data_dir, "new_merchant_transactions_train.csv"),
                                         parse_dates=["purchase_date"])
        new_trans_df_train = compress_int(new_trans_df_train)
        send_msg("load done")
        return train_df, hist_df_train, new_trans_df_train
        pass

    print("loading test.csv ...")
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"), parse_dates=["first_active_month"], dtype=d)
    test_df.info(memory_usage='deep')

    print("loading historical_transactions.csv ...")
    hist_df = pd.read_csv(os.path.join(data_dir, "historical_transactions.csv"), parse_dates=["purchase_date"])
    print('    compressing ...')
    hist_df = compressByDType(hist_df)
    print('    split to get train hist ...')
    hist_df_train = hist_df[hist_df.card_id.isin(set(train_df['card_id'].unique()))]
    print('    saving ... ')
    hist_df_train.to_csv(os.path.join(data_dir, "historical_transactions_train.csv"))
    print('    split to get test hist ...')
    hist_df_test = hist_df[hist_df.card_id.isin(set(test_df['card_id'].unique()))]
    print('    saving ... ')
    hist_df_test.to_csv(os.path.join(data_dir, "historical_transactions_test.csv"))
    del hist_df_test
    del hist_df
    gc.collect()

    print("loading new_merchant_transactions.csv ...")
    new_trans_df = pd.read_csv(os.path.join(data_dir, "new_merchant_transactions.csv"),
                               parse_dates=["purchase_date"])
    print('    compressing ...')
    new_trans_df = compressByDType(new_trans_df)
    print('    split to get train new trans ...')
    new_trans_df_train = new_trans_df[new_trans_df.card_id.isin(set(train_df['card_id'].unique()))]
    print('    saving ... ')
    new_trans_df_train.to_csv(os.path.join(data_dir, "new_merchant_transactions_train.csv"))
    print('    split to get test new trans ...')
    new_trans_df_test = new_trans_df[new_trans_df.card_id.isin(set(test_df['card_id'].unique()))]
    print('    saving ... ')
    new_trans_df_test.to_csv(os.path.join(data_dir, "new_merchant_transactions_test.csv"))
    del new_trans_df_test
    del new_trans_df
    gc.collect()

    send_msg("split and save done")
    return train_df, hist_df_train, new_trans_df_train


def agg2(df_train, df_test, df_hist_trans):
    aggs = {}
    for col in ['month', 'hour', 'weekofyear', 'dayofweek', 'year', 'subsector_id', 'merchant_category_id']:
        aggs[col] = ['nunique']

    aggs['purchase_amount'] = ['sum', 'max', 'min', 'mean', 'var']
    aggs['installments'] = ['sum', 'max', 'min', 'mean', 'var']
    aggs['purchase_date'] = ['max', 'min']
    aggs['month_lag'] = ['max', 'min', 'mean', 'var']
    aggs['month_diff'] = ['mean']
    aggs['authorized_flag'] = ['sum', 'mean']
    aggs['weekend'] = ['sum', 'mean']
    aggs['category_1'] = ['sum', 'mean']
    aggs['card_id'] = ['size']

    for col in ['category_2', 'category_3']:
        df_hist_trans[col + '_mean'] = df_hist_trans.groupby([col])['purchase_amount'].transform('mean')
        aggs[col + '_mean'] = ['mean']

    new_columns = get_new_columns('hist', aggs)
    df_hist_trans_group = df_hist_trans.groupby('card_id').agg(aggs)
    df_hist_trans_group.columns = new_columns
    df_hist_trans_group.reset_index(drop=False, inplace=True)
    df_hist_trans_group['hist_purchase_date_diff'] = (
            df_hist_trans_group['hist_purchase_date_max'] - df_hist_trans_group['hist_purchase_date_min']).dt.days
    df_hist_trans_group['hist_purchase_date_average'] = df_hist_trans_group['hist_purchase_date_diff'] / \
                                                        df_hist_trans_group['hist_card_id_size']
    df_hist_trans_group['hist_purchase_date_uptonow'] = (
            datetime.datetime.today() - df_hist_trans_group['hist_purchase_date_max']).dt.days
    df_train = df_train.merge(df_hist_trans_group, on='card_id', how='left')
    df_test = df_test.merge(df_hist_trans_group, on='card_id', how='left')
    del df_hist_trans_group
    gc.collect()

    return df_train, df_test


def get_new_columns(name, aggs):
    return [name + '_' + k + '_' + agg for k in aggs.keys() for agg in aggs[k]]


def compressByDType(df_new_merchant_trans):
    """

    :param df_new_merchant_trans:
    :return:
    """
    df_new_merchant_trans = df_new_merchant_trans.drop(columns=['merchant_id'])

    df_new_merchant_trans['category_2'].fillna(1.0, inplace=True)
    df_new_merchant_trans['category_3'].fillna('D', inplace=True)
    df_new_merchant_trans['authorized_flag'].fillna('Y', inplace=True)

    df_new_merchant_trans['authorized_flag'] = df_new_merchant_trans['authorized_flag'].map({'Y': 1, 'N': 0})
    df_new_merchant_trans['category_1'] = df_new_merchant_trans['category_1'].map({'Y': 1, 'N': 0})
    df_new_merchant_trans['category_3'] = df_new_merchant_trans['category_3'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3})

    df_new_merchant_trans['category_1'] = pd.to_numeric(df_new_merchant_trans['category_1'], downcast='integer')
    df_new_merchant_trans['category_2'] = pd.to_numeric(df_new_merchant_trans['category_2'], downcast='integer')
    df_new_merchant_trans['category_3'] = pd.to_numeric(df_new_merchant_trans['category_3'], downcast='integer')

    df_new_merchant_trans['merchant_category_id'] = pd.to_numeric(df_new_merchant_trans['merchant_category_id'],
                                                                  downcast='integer')
    df_new_merchant_trans['authorized_flag'] = pd.to_numeric(df_new_merchant_trans['authorized_flag'],
                                                             downcast='integer')
    df_new_merchant_trans['city_id'] = pd.to_numeric(df_new_merchant_trans['city_id'], downcast='integer')
    df_new_merchant_trans['installments'] = pd.to_numeric(df_new_merchant_trans['installments'], downcast='integer')

    df_new_merchant_trans['state_id'] = pd.to_numeric(df_new_merchant_trans['state_id'], downcast='integer')
    df_new_merchant_trans['subsector_id'] = pd.to_numeric(df_new_merchant_trans['subsector_id'], downcast='integer')
    df_new_merchant_trans['month_lag'] = pd.to_numeric(df_new_merchant_trans['month_lag'], downcast='integer')

    df_new_merchant_trans['purchase_date'] = pd.to_datetime(df_new_merchant_trans['purchase_date'])

    df_new_merchant_trans['year'] = df_new_merchant_trans['purchase_date'].dt.year
    df_new_merchant_trans['weekofyear'] = df_new_merchant_trans['purchase_date'].dt.weekofyear
    df_new_merchant_trans['month'] = df_new_merchant_trans['purchase_date'].dt.month
    df_new_merchant_trans['dayofweek'] = df_new_merchant_trans['purchase_date'].dt.dayofweek
    df_new_merchant_trans['weekend'] = (df_new_merchant_trans.purchase_date.dt.weekday >= 5).astype(int)
    df_new_merchant_trans['hour'] = df_new_merchant_trans['purchase_date'].dt.hour
    # https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/73244
    df_new_merchant_trans['month_diff'] = ((datetime.datetime.today() - df_new_merchant_trans[
        'purchase_date']).dt.days) // 30
    df_new_merchant_trans['month_diff'] += df_new_merchant_trans['month_lag']

    df_new_merchant_trans['weekofyear'] = pd.to_numeric(df_new_merchant_trans['weekofyear'], downcast='integer')
    df_new_merchant_trans['month'] = pd.to_numeric(df_new_merchant_trans['month'], downcast='integer')
    df_new_merchant_trans['dayofweek'] = pd.to_numeric(df_new_merchant_trans['dayofweek'], downcast='integer')
    df_new_merchant_trans['weekend'] = pd.to_numeric(df_new_merchant_trans['weekend'], downcast='integer')
    df_new_merchant_trans['hour'] = pd.to_numeric(df_new_merchant_trans['hour'], downcast='integer')
    df_new_merchant_trans['month_diff'] = pd.to_numeric(df_new_merchant_trans['month_diff'], downcast='integer')

    df_new_merchant_trans.info(memory_usage='deep')

    return df_new_merchant_trans


def read_data_c2(train_df,
                 test_df,
                 hist_df,
                 new_trans_df):
    target = train_df['target']
    del train_df['target']

    es_train = ft.EntitySet(id='es_train')
    es_test = ft.EntitySet(id='es_test')

    es_train = es_train.entity_from_dataframe(entity_id='train', dataframe=train_df,
                                              index='client_id', time_index='joined')
    es_train = es_train.entity_from_dataframe(entity_id='history', dataframe=hist_df,
                                              index='', time_index='purchase_date')
    es_train = es_train.entity_from_dataframe(entity_id='new_trans', dataframe=new_trans_df,
                                              index='', time_index='purchase_date')
    # Relationship between clients and previous loans
    r_client_previous = ft.Relationship(es_train['train']['card_id'],
                                        es_train['history']['card_id'])

    # Add the relationship to the entity set
    es_train = es_train.add_relationship(r_client_previous)
    r_client_previous = ft.Relationship(es_train['train']['card_id'],
                                        es_train['new_trans']['card_id'])

    # Add the relationship to the entity set
    es_train = es_train.add_relationship(r_client_previous)
    x_train, feature_names = ft.dfs(entityset=es_train, target_entity='train',
                                    max_depth=2)

    es_test = es_test.entity_from_dataframe(entity_id='test', dataframe=train_df,
                                            index='client_id', time_index='joined')
    es_test = es_test.entity_from_dataframe(entity_id='history', dataframe=hist_df,
                                            index='', time_index='purchase_date')
    es_test = es_test.entity_from_dataframe(entity_id='new_trans', dataframe=new_trans_df,
                                            index='', time_index='purchase_date')
    # Relationship between clients and previous loans
    r_client_previous = ft.Relationship(es_test['test']['card_id'],
                                        es_test['history']['card_id'])

    # Add the relationship to the entity set
    es_test = es_test.add_relationship(r_client_previous)
    r_client_previous = ft.Relationship(es_test['test']['card_id'],
                                        es_test['new_trans']['card_id'])

    # Add the relationship to the entity set
    es_test = es_test.add_relationship(r_client_previous)
    x_test, feature_names = ft.dfs(entityset=es_test, target_entity='test',
                                   max_depth=2)

    return x_train, target, x_test


def read_data_c(train_df,
                test_df,
                hist_df,
                new_trans_df,
                version='c1.0'):
    """

    :param train_df:
    :param test_df:
    :param hist_df:
    :param new_trans_df:
    :param version:
    :return:
    """
    # 0. compress
    print("compressing ... ")
    hist_df = compressByDType(hist_df)
    new_trans_df = compressByDType(new_trans_df)
    print("compressing done")
    if version == 'c2.0':
        return read_data_c2(train_df,
                            test_df,
                            hist_df,
                            new_trans_df)

    # 1. [整合成一个data frame] merger them as one df
    print("agg ...")
    agg2(train_df, test_df, hist_df)
    agg2(train_df, test_df, new_trans_df)

    del hist_df
    gc.collect()
    del new_trans_df
    gc.collect()

    train_df['outliers'] = 0
    train_df.loc[train_df['target'] < -30, 'outliers'] = 1
    train_df['outliers'].value_counts()

    for df in [train_df, test_df]:
        df['first_active_month'] = pd.to_datetime(df['first_active_month'])
        df['dayofweek'] = df['first_active_month'].dt.dayofweek
        df['weekofyear'] = df['first_active_month'].dt.weekofyear
        df['month'] = df['first_active_month'].dt.month
        df['elapsed_time'] = (datetime.datetime.today() - df['first_active_month']).dt.days
        df['hist_first_buy'] = (df['hist_purchase_date_min'] - df['first_active_month']).dt.days
        df['new_hist_first_buy'] = (df['new_hist_purchase_date_min'] - df['first_active_month']).dt.days
        for f in ['hist_purchase_date_max', 'hist_purchase_date_min', 'new_hist_purchase_date_max', \
                  'new_hist_purchase_date_min']:
            df[f] = df[f].astype(np.int64) * 1e-9
        df['card_id_total'] = df['new_hist_card_id_size'] + df['hist_card_id_size']
        df['purchase_amount_total'] = df['new_hist_purchase_amount_sum'] + df['hist_purchase_amount_sum']

    for f in ['feature_1', 'feature_2', 'feature_3']:
        order_label = train_df.groupby([f])['outliers'].mean()
        train_df[f] = train_df[f].map(order_label)
        test_df[f] = test_df[f].map(order_label)

    target = train_df['target']
    del train_df['target']

    return train_df, test_df, target


def read_data_c1(train_df,
                 test_df,
                 hist_df,
                 new_trans_df):
    pass

# train_df, hist_df_train, new_trans_df_train = split_trans_into_train_test()
