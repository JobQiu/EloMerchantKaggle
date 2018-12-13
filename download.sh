#!/usr/bin/env bash

DATA_DIR=/content/EloMerchantKaggle/data
mkdir $DATA_DIR
pip install kaggle
pip install -U featuretools

rm /root/.kaggle/kaggle.json

# download the API credentials
wget https://raw.githubusercontent.com/JobQiu/EloMerchantKaggle/master/data/kaggle.json -P /root/.kaggle/
chmod 600 /root/.kaggle/kaggle.json
#

kaggle competitions download -c elo-merchant-category-recommendation -p /content/EloMerchantKaggle/data


unzip /content/EloMerchantKaggle/data/historical_transactions.csv.zip -d /content/EloMerchantKaggle/data
unzip /content/EloMerchantKaggle/data/merchants.csv.zip -d /content/EloMerchantKaggle/data
unzip /content/EloMerchantKaggle/data/new_merchant_transactions.csv.zip -d /content/EloMerchantKaggle/data
unzip /content/EloMerchantKaggle/data/sample_submission.csv.zip -d /content/EloMerchantKaggle/data
unzip /content/EloMerchantKaggle/data/test.csv.zip -d /content/EloMerchantKaggle/data
unzip /content/EloMerchantKaggle/data/train.csv.zip -d /content/EloMerchantKaggle/data

