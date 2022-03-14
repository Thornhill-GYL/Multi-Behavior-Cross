# coding: utf-8
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from config import FRAC

if __name__ == "__main__":

    user = pd.read_csv('data/user_profile.csv')
    print('user_head:\n')
    print(user.head())
    sample = pd.read_csv('data/raw_sample.csv')
    print('sample_head:\n')
    print(sample.head())

    if not os.path.exists('sampled_data/'):
        os.mkdir('sampled_data/')

    if os.path.exists('sampled_data/user_profile_' + str(FRAC) + '_.pkl') and os.path.exists(
            'sampled_data/raw_sample_' + str(FRAC) + '_.pkl'):
        user_sub = pd.read_pickle(
            'sampled_data/user_profile_' + str(FRAC) + '_.pkl')
        sample_sub = pd.read_pickle(
            'sampled_data/raw_sample_' + str(FRAC) + '_.pkl')
    else:

        if FRAC < 1.0:
            user_sub = user.sample(frac=FRAC, random_state=1024)
        else:
            user_sub = user
            
        sample_sub = sample.loc[sample.user.isin(user_sub.userid.unique())]
        pd.to_pickle(user_sub, 'sampled_data/user_profile_' +
                     str(FRAC) + '.pkl')
        pd.to_pickle(sample_sub, 'sampled_data/raw_sample_' +
                     str(FRAC) + '.pkl')

    if os.path.exists('sampled_data/behavior_log_pv.pkl'):#假设如果存在pv的，那么其他特征的文件也应该相关存在
        log_pv = pd.read_pickle('sampled_data/behavior_log_pv.pkl')
        log_purse = pd.read_pickle('sampled_data/behavior_log_purse.pkl')
        log_cart = pd.read_pickle('sampled_data/behavior_log_cart.pkl')
    else:
        log_all = pd.read_csv('data/behavior_log.csv')
        
        log_pv = log_all.loc[log_all['btag'] == 'pv']
        log_purse = log_all.loc[log_all['btag']=='buy']
        log_cart = log_all.loc[log_all['btag']=='cart']
        pd.to_pickle(log_pv, 'sampled_data/behavior_log_pv.pkl')#所有的先保存到pkl
        pd.to_pickle(log_purse, 'sampled_data/behavior_log_purse.pkl')
        pd.to_pickle(log_cart, 'sampled_data/behavior_log_cart.pkl')


    userset = user_sub.userid.unique()
    log_pv = log_pv.loc[log_pv.user.isin(userset)]
    log_purse = log_purse.loc[log_purse.user.isin(userset)]
    log_cart = log_cart.loc[log_cart.user.isin(userset)]
    print('log_pv_head:\n')
    print(log_pv.head())
    # pd.to_pickle(log, 'sampled_data/behavior_log_pv_user_filter_' + str(FRAC) + '_.pkl')

    ad = pd.read_csv('data/ad_feature.csv')
    ad['brand'] = ad['brand'].fillna(-1)

    lbe = LabelEncoder()
    # unique_cate_id = ad['cate_id'].unique()
    # log = log.loc[log.cate.isin(unique_cate_id)]

    unique_cate_id = np.concatenate(
        (ad['cate_id'].unique(), log_pv['cate'].unique(),log_purse['cate'].unique(),log_cart['cate'].unique()))

    lbe.fit(unique_cate_id)
    ad['cate_id'] = lbe.transform(ad['cate_id']) + 1
    print('origin_cate:\n')
    print(log_pv['cate'])
    log_pv['cate'] = lbe.transform(log_pv['cate']) + 1
    print('trans_cate:\n')
    print(log_pv['cate'])
    log_purse['cate']=lbe.transform(log_purse['cate'])+1
    log_cart['cate'] = lbe.transform(log_cart['cate'])+1

    lbe = LabelEncoder()
    # unique_brand = np.ad['brand'].unique()
    # log = log.loc[log.brand.isin(unique_brand)]

    unique_brand = np.concatenate(
        (ad['brand'].unique(), log_pv['brand'].unique(),log_purse['brand'].unique(),log_cart['brand'].unique()))

    lbe.fit(unique_brand)
    ad['brand'] = lbe.transform(ad['brand']) + 1
    log_pv['brand'] = lbe.transform(log_pv['brand']) + 1
    log_purse['brand'] = lbe.transform(log_purse['brand'])+1
    log_cart['brand'] = lbe.transform(log_cart['brand'])+1

    #log = log.loc[log.user.isin(sample_sub.user.unique())]
    log_pv = log_pv.loc[log_pv.user.isin(sample_sub.user.unique())]#重复确认了单一性
    log_purse = log_purse.loc[log_purse.user.isin(sample_sub.user.unique())]
    log_cart = log_cart.loc[log_cart.user.isin(sample_sub.user.unique())]

    #log.drop(columns=['btag'], inplace=True)
    log_pv.drop(columns = ['btag'],inplace = True)
    log_cart.drop(columns = ['btag'],inplace = True)
    log_purse.drop(columns = ['btag'],inplace = True)

    #log = log.loc[log['time_stamp'] > 0]

    log_pv = log_pv.loc[log_pv['time_stamp']>0]
    log_cart = log_cart.loc[log_cart['time_stamp']>0]
    log_purse = log_purse.loc[log_purse['time_stamp']>0]


    pd.to_pickle(ad, 'sampled_data/ad_feature_enc_' + str(FRAC) + '.pkl')
    pd.to_pickle(
        log_pv, 'sampled_data/behavior_log_pv_user_filter_enc_' + str(FRAC) + '.pkl')
    pd.to_pickle(
        log_purse, 'sampled_data/behavior_log_purse_user_filter_enc_' + str(FRAC) + '.pkl')
    pd.to_pickle(
        log_cart, 'sampled_data/behavior_log_cart_user_filter_enc_' + str(FRAC) + '.pkl')

    print("0_gen_sampled_data done")