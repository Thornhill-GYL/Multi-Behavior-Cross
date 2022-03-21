# coding: utf-8
import gc

import pandas as pd
from joblib import Parallel, delayed

from config import FRAC

def gen_session_list(uid, t, interval_time=None):#action不需要时间间隔
    t.sort_values('time_stamp', inplace=True, ascending=True)
    last_time = 1483574401  # pd.to_datetime("2017-01-05 00:00:01")
    session_list = []
    session = []
    for row in t.iterrows():
        time_stamp = row[1]['time_stamp']
        # pd_time = pd.to_datetime(timestamp_datetime(time_stamp))
        delta = time_stamp - last_time
        cate_id = row[1]['cate']
        brand = row[1]['brand']
        # delta.total_seconds()
        if delta > interval_time:  # Session begin when current behavior and the last behavior are separated by more than interval_time.
            if len(session) > 2:  # Only use sessions that have >2 behaviors
                session_list.append(session[:])
            session = []

        session.append((cate_id, brand, time_stamp))
        last_time = time_stamp
    if len(session) > 2:
        session_list.append(session[:])
    return uid, session_list

"""
session list的结果
{29: [[(5257, 263055, 1493801016),
   (5257, 23899, 1493803796),
   (4914, 291316, 1493864492),
   (5587, 21436, 1493886986),
   (5587, 21436, 1493888468),
   (5587, 21436, 1493888468),
   (5587, 21436, 1493888509),
   (5587, 21436, 1493888509),
   (3686, 44211, 1493889138),
   (3686, 44211, 1493889223),
   (5491, 356945, 1493962714)],
  [(3925, 34611, 1494319091),
   (3925, 34611, 1494319091),
   (3784, 191738, 1494324093)]],
 36: [],
 39: [],
 43: [[(9962, 52812, 1494032029),
   (9962, 52812, 1494032029),
   (6263, 54782, 1494032029),
   (6263, 54782, 1494032029),
   (6501, 132219, 1494177707),
   (6501, 132219, 1494177707),
   (6501, 132219, 1494177707),
   (9962, 54782, 1494301187),
   (7320, 23321, 1494301187),
   (9962, 54782, 1494301187),
   (9961, 54782, 1494301187),
   (5809, 54782, 1494301361),
   (5809, 54782, 1494301361),
   (5809, 54782, 1494301361),
   (5809, 54782, 1494301361),
   (5809, 54782, 1494301864),
   (7320, 23321, 1494301864),
   (5809, 54782, 1494301864),
   (9961, 54782, 1494301864),
   (9962, 54782, 1494301864),
   (9962, 54782, 1494301864),
   (5809, 54782, 1494301864),
   (5809, 54782, 1494301864),
   (9961, 52812, 1494302050),
   (7320, 23321, 1494302413),
   (255, 196664, 1494472252),
   (7319, 196664, 1494472252),
   (7319, 196664, 1494472252),
   (5144, 24983, 1494573363),
   (4542, 24983, 1494573363)]]
   """

def gen_action_list(uid, t, interval_time = None):
    t.sort_values('time_stamp', inplace=True, ascending=True)
    action_list = []
    action = []
    for row in t.iterrows():
        time_stamp = row[1]['time_stamp']
        # pd_time = pd.to_datetime(timestamp_datetime())
        # delta = pd_time  - last_time
        cate_id = row[1]['cate']
        brand = row[1]['brand']
        action.append((cate_id, brand, time_stamp))

    if len(action) > 2:
        action_list.append(action[:])
    return uid, action_list
"""
action_list 可以发现user43的action没有收到时间的裁剪
29: [[(5257, 263055, 1493801016),
   (5257, 23899, 1493803796),
   (4914, 291316, 1493864492),
   (5587, 21436, 1493886986),
   (5587, 21436, 1493888468),
   (5587, 21436, 1493888468),
   (5587, 21436, 1493888509),
   (5587, 21436, 1493888509),
   (3686, 44211, 1493889138),
   (3686, 44211, 1493889223),
   (5491, 356945, 1493962714),
   (3925, 34611, 1494319091),
   (3925, 34611, 1494319091),
   (3784, 191738, 1494324093)]],
 36: [],
 39: [],
 43: [[(5175, 248781, 1493819618),
   (9962, 52812, 1494032029),
   (9962, 52812, 1494032029),
   (6263, 54782, 1494032029),
   (6263, 54782, 1494032029),
   (6501, 132219, 1494177707),
   (6501, 132219, 1494177707),
   (6501, 132219, 1494177707),
   (9962, 54782, 1494301187),
   (7320, 23321, 1494301187),
   (9962, 54782, 1494301187),
   (9961, 54782, 1494301187),
   (5809, 54782, 1494301361),
   (5809, 54782, 1494301361),
   (5809, 54782, 1494301361),
   (5809, 54782, 1494301361),
   (5809, 54782, 1494301864),
   (7320, 23321, 1494301864),
   (5809, 54782, 1494301864),
   (9961, 54782, 1494301864),
   (9962, 54782, 1494301864),
   (9962, 54782, 1494301864),
   (5809, 54782, 1494301864),
   (5809, 54782, 1494301864),
   (9961, 52812, 1494302050),
   (7320, 23321, 1494302413),
   (255, 196664, 1494472252),
   (7319, 196664, 1494472252),
   (7319, 196664, 1494472252),
   (5144, 24983, 1494573363),
   (4542, 24983, 1494573363)]],
"""

def applyParallel(df_grouped, func, n_jobs,backend='multiprocessing',interval_time=None):
    """Use Parallel and delayed """  # backend='threading'
    results = Parallel(n_jobs=n_jobs, verbose=4, backend=backend)(
        delayed(func)(name, group,interval_time) for name, group in df_grouped)

    return {k: v for k, v in results}

def gen_user_hist_sessions(segmentation,btag,FRAC=0.25,interval_time=None):
    if segmentation not in ['action', 'session']:
        raise ValueError('segmentation method must be action or session')

    print("gen " + segmentation + " hist", FRAC)
    
    name = 'sampled_data/behavior_log_'+btag+'_user_filter_enc_' + str(FRAC) + '.csv'
    data = pd.read_csv(name)
    data = data.loc[data.time_stamp >= 1493769600]  # 0503-0513
    # 0504~1493856000
    # 0503 1493769600

    user = pd.read_pickle('sampled_data/user_profile_' + str(FRAC) + '.pkl')

    n_samples = user.shape[0]
    print(n_samples)
    batch_size = 150000
    iters = (n_samples - 1) // batch_size + 1

    print("total", iters, "iters", "batch_size", batch_size)
    for i in range(0, iters):
        target_user = user['userid'].values[i * batch_size:(i + 1) * batch_size]
        sub_data = data.loc[data.user.isin(target_user)]
        print(i, 'iter start')
        df_grouped = sub_data.groupby('user')
        if segmentation == 'action':
            user_hist_session = applyParallel(
                df_grouped, gen_action_list, n_jobs=20, backend='loky')
            print('user_hist_action_done\n')
        else:
            user_hist_session = applyParallel(
                df_grouped, gen_session_list, n_jobs=20, backend='multiprocessing',interval_time=interval_time)
            print('user_hist_session_done\n')
        pd.to_pickle(user_hist_session, 'sampled_data/user_hist_session_' +btag+
                     str(FRAC) + '_' + segmentation + '_' + str(i) + '.pkl')
        print(i, 'pickled')
        del user_hist_session
        gc.collect()
        print(i, 'del')

    print("1_gen " + segmentation + " hist sess done")
    
if __name__ == "__main__":
    btags=['pv','purse','cart']
    interval_time=[1800,172800,43200]#不同的间隔切分
    for btag, time in zip(btags,interval_time):
        gen_user_hist_sessions('action', btag,FRAC)
        gen_user_hist_sessions('session',btag,FRAC,time)

    