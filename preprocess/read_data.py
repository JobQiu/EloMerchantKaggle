#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 16:32:58 2018

@author: 
"""
import os
import pandas as pd
import numpy as np

def read_data(data_dir="/content/EloMerchantKaggle/data"):
    """
    
    """
    
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"), parse_dates=["first_active_month"])
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"), parse_dates=["first_active_month"])
    hist_df = pd.read_csv(os.path.join(data_dir, "historical_transactions.csv"))
    
    target_col = "target"
    
    gdf = hist_df.groupby("card_id")
    gdf = gdf["purchase_amount"].size().reset_index()
    gdf.columns = ["card_id", "num_hist_transactions"]
    train_df = pd.merge(train_df, gdf, on="card_id", how="left")
    test_df = pd.merge(test_df, gdf, on="card_id", how="left")  
    bins = [0, 10, 20, 30, 40, 50, 75, 100, 150, 200, 500, 10000]
    train_df['binned_num_hist_transactions'] = pd.cut(train_df['num_hist_transactions'], bins)
    
    gdf = hist_df.groupby("card_id")
    gdf = gdf["purchase_amount"].agg(['sum', 'mean', 'std', 'min', 'max']).reset_index()
    gdf.columns = ["card_id", "sum_hist_trans", "mean_hist_trans", "std_hist_trans", "min_hist_trans", "max_hist_trans"]
    train_df = pd.merge(train_df, gdf, on="card_id", how="left")
    test_df = pd.merge(test_df, gdf, on="card_id", how="left")  
    
    bins = np.percentile(train_df["sum_hist_trans"], range(0,101,10))
    train_df['binned_sum_hist_trans'] = pd.cut(train_df['sum_hist_trans'], bins)
    
    bins = np.percentile(train_df["mean_hist_trans"], range(0,101,10))
    train_df['binned_mean_hist_trans'] = pd.cut(train_df['mean_hist_trans'], bins)

    new_trans_df = pd.read_csv(os.path.join(data_dir, "new_merchant_transactions.csv"))
    gdf = new_trans_df.groupby("card_id")
    gdf = gdf["purchase_amount"].size().reset_index()
    gdf.columns = ["card_id", "num_merch_transactions"]
    train_df = pd.merge(train_df, gdf, on="card_id", how="left")
    test_df = pd.merge(test_df, gdf, on="card_id", how="left")
    
    bins = [0, 10, 20, 30, 40, 50, 75, 10000]
    train_df['binned_num_merch_transactions'] = pd.cut(train_df['num_merch_transactions'], bins)
    
    gdf = new_trans_df.groupby("card_id")
    gdf = gdf["purchase_amount"].agg(['sum', 'mean', 'std', 'min', 'max']).reset_index()
    gdf.columns = ["card_id", "sum_merch_trans", "mean_merch_trans", "std_merch_trans", "min_merch_trans", "max_merch_trans"]
    train_df = pd.merge(train_df, gdf, on="card_id", how="left")
    test_df = pd.merge(test_df, gdf, on="card_id", how="left")
    
    
    bins = np.nanpercentile(train_df["sum_merch_trans"], range(0,101,10))
    train_df['binned_sum_merch_trans'] = pd.cut(train_df['sum_merch_trans'], bins)
    
    bins = np.nanpercentile(train_df["mean_merch_trans"], range(0,101,10))
    train_df['binned_mean_merch_trans'] = pd.cut(train_df['mean_merch_trans'], bins)
    
    train_df["year"] = train_df["first_active_month"].dt.year
    test_df["year"] = test_df["first_active_month"].dt.year
    train_df["month"] = train_df["first_active_month"].dt.month
    test_df["month"] = test_df["first_active_month"].dt.month
    
    cols_to_use = ["feature_1", "feature_2", "feature_3", "year", "month", 
                   "num_hist_transactions", "sum_hist_trans", "mean_hist_trans", "std_hist_trans", 
                   "min_hist_trans", "max_hist_trans",
                   "num_merch_transactions", "sum_merch_trans", "mean_merch_trans", "std_merch_trans",
                   "min_merch_trans", "max_merch_trans",
                  ]
    
    
    return  train_df[cols_to_use],  test_df[cols_to_use],  train_df[target_col].values

if __name__ == "__main__":
    x_train, y_train, x_test = read_data(data_dir ="/Users/xavier.qiu/Documents/Kaggle/EloMerchantKaggle/data" )
    
    #%%
