
Auto-feature engineering? 



## train.csv

## test.csv

![](https://ws3.sinaimg.cn/large/006tNbRwly1fy3wb5buthj30d40920tb.jpg)

There is one sample whose first_active_month is None

## historical_transactions.csv & new_merchant_transactions.csv

![](https://ws2.sinaimg.cn/large/006tNbRwly1fy3w9ize0hj30hi0ikmz7.jpg)

```
for df in [df_hist_trans,df_new_merchant_trans]:
    df['category_2'].fillna(1.0,inplace=True)
    df['category_3'].fillna('A',inplace=True)
    df['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)
```



## merchants.csv
