#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 16:32:58 2018

@author:
"""
import os
import pandas as pd
import numpy as np
from preprocess.read_data_congmin import read_data_c

def read_data(data_dir="/content/EloMerchantKaggle/data",
            version = 'default'):
    """
    """
    print("loading train.csv ..." )
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"), parse_dates=["first_active_month"])

    print("loading test.csv ..." )
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"), parse_dates=["first_active_month"])

    print("loading historical_transactions.csv ..." )
    hist_df = pd.read_csv(os.path.join(data_dir, "historical_transactions.csv"))

    print("loading new_merchant_transactions.csv ..." )
    new_trans_df = pd.read_csv(os.path.join(data_dir, "new_merchant_transactions.csv"))
    
    if version.startswith('c'):
        return read_data_c(train_df, test_df, hist_df, new_trans_df, version)

    target_col = "target"

    train_df_stat, test_df_stat = addStatFeatures(train_df,test_df,hist_df,new_trans_df)

    if version != '' and version.strip()[0] == 'c':
        pass
    elif version != '' and version.strip()[0] == 'z':
        # Feature engineering: polynomial features
        poly_features = train_df_stat[['feature_1','feature_2','feature_3']]
        poly_features_test = test_df_stat[['feature_1','feature_2','feature_3']]
        poly_target = train_df_stat[['target']]
        from sklearn.preprocessing import PolynomialFeatures
        poly_transform = PolynomialFeatures(degree=3)
        poly_transform.fit(poly_features)
        poly_transform.fit(poly_features_test)

        poly_features = poly_transform.transform(poly_features)
        poly_features_test = poly_transform.transform(poly_features_test)

        # Create a dataframe of the features 
        poly_features = pd.DataFrame(poly_features, 
                                     columns = poly_transform.get_feature_names(['feature_1', 'feature_2', 
                                                                                   'feature_3']))
        poly_features_test = pd.DataFrame(poly_features,
                                     columns= poly_transform.get_feature_names(['feature_1', 'feature_2', 
                                                                                   'feature_3']))

        poly_features['card_id'] = train_df_stat['card_id']
        poly_features_test['card_id'] = test_df_stat['card_id']

        train_df_poly = pd.merge(train_df_stat,poly_features,on='card_id',how='left')
        test_df_poly = pd.merge(test_df_stat,poly_features_test,on='card_id',how='left')

        cols_to_use = [ "feature_1_x", "feature_2_x", "feature_3_x","year", "month",
                       "num_hist_transactions", "sum_hist_trans", "mean_hist_trans", "std_hist_trans",
                       "min_hist_trans", "max_hist_trans",
                       "num_merch_transactions", "sum_merch_trans", "mean_merch_trans", "std_merch_trans",
                       "min_merch_trans", "max_merch_trans",
                      ]
        print(cols_to_use)
        cols_to_use.extend(poly_transform.get_feature_names(['feature_1', 'feature_2', 'feature_3'])[4:])                                                              
        
        print(cols_to_use)
        print(train_df_poly[:10])
        return train_df_poly[cols_to_use] ,test_df_poly[cols_to_use], train_df_poly[target_col].values

    else:
        # default only use statistic features
        cols_to_use = ["feature_1", "feature_2", "feature_3", "year", "month",
                       "num_hist_transactions", "sum_hist_trans", "mean_hist_trans", "std_hist_trans",
                       "min_hist_trans", "max_hist_trans",
                       "num_merch_transactions", "sum_merch_trans", "mean_merch_trans", "std_merch_trans",
                       "min_merch_trans", "max_merch_trans",
                      ]
        return train_df_stat[cols_to_use],  test_df_stat[cols_to_use],  train_df_stat[target_col].values

# add additional statistical features 
def addStatFeatures(train_df,test_df,hist_df,new_trans_df):
    train_df_stat = train_df.copy(deep = True)
    test_df_stat = test_df.copy(deep=True)

    gdf = hist_df.groupby("card_id")
    gdf = gdf["purchase_amount"].size().reset_index()
    gdf.columns = ["card_id", "num_hist_transactions"]
    train_df_stat = pd.merge(train_df_stat, gdf, on="card_id", how="left")
    test_df_stat = pd.merge(test_df_stat, gdf, on="card_id", how="left")
    bins = [0, 10, 20, 30, 40, 50, 75, 100, 150, 200, 500, 10000]
    train_df_stat['binned_num_hist_transactions'] = pd.cut(train_df_stat['num_hist_transactions'], bins)

    gdf = hist_df.groupby("card_id")
    gdf = gdf["purchase_amount"].agg(['sum', 'mean', 'std', 'min', 'max']).reset_index()
    gdf.columns = ["card_id", "sum_hist_trans", "mean_hist_trans", "std_hist_trans", "min_hist_trans", "max_hist_trans"]
    train_df_stat = pd.merge(train_df_stat, gdf, on="card_id", how="left")
    test_df_stat = pd.merge(test_df_stat, gdf, on="card_id", how="left")

    bins = np.percentile(train_df_stat["sum_hist_trans"], range(0,101,10))
    train_df_stat['binned_sum_hist_trans'] = pd.cut(train_df_stat['sum_hist_trans'], bins)

    bins = np.percentile(train_df_stat["mean_hist_trans"], range(0,101,10))
    train_df_stat['binned_mean_hist_trans'] = pd.cut(train_df_stat['mean_hist_trans'], bins)


    gdf = new_trans_df.groupby("card_id")
    gdf = gdf["purchase_amount"].size().reset_index()
    gdf.columns = ["card_id", "num_merch_transactions"]
    train_df_stat = pd.merge(train_df_stat, gdf, on="card_id", how="left")
    test_df_stat = pd.merge(test_df_stat, gdf, on="card_id", how="left")

    bins = [0, 10, 20, 30, 40, 50, 75, 10000]
    train_df_stat['binned_num_merch_transactions'] = pd.cut(train_df_stat['num_merch_transactions'], bins)

    gdf = new_trans_df.groupby("card_id")
    gdf = gdf["purchase_amount"].agg(['sum', 'mean', 'std', 'min', 'max']).reset_index()
    gdf.columns = ["card_id", "sum_merch_trans", "mean_merch_trans", "std_merch_trans", "min_merch_trans", "max_merch_trans"]
    train_df_stat = pd.merge(train_df_stat, gdf, on="card_id", how="left")
    test_df_stat = pd.merge(test_df_stat, gdf, on="card_id", how="left")

    bins = np.nanpercentile(train_df_stat["sum_merch_trans"], range(0,101,10))
    train_df_stat['binned_sum_merch_trans'] = pd.cut(train_df_stat['sum_merch_trans'], bins)
    bins = np.nanpercentile(train_df_stat["mean_merch_trans"], range(0,101,10))
    train_df_stat['binned_mean_merch_trans'] = pd.cut(train_df_stat['mean_merch_trans'], bins)

    train_df_stat["year"] = train_df_stat["first_active_month"].dt.year
    test_df_stat["year"] = test_df_stat["first_active_month"].dt.year
    train_df_stat["month"] = train_df_stat["first_active_month"].dt.month
    test_df_stat["month"] = test_df_stat["first_active_month"].dt.month

    return train_df_stat, test_df_stat

if __name__ == "__main__":
    x_train, y_train, x_test = read_data(data_dir ="/Users/xavier.qiu/Documents/Kaggle/EloMerchantKaggle/data")
#%%
