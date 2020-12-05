import sys
import pandas as pd
import numpy as np
np.random.seed(1)
from sklearn.decomposition import PCA

SECONDS_PER_DAY = 86400
SECONDS_PER_HOUR = 60 * 60
TRAIN_DATA_DAYS = 21  # the length of the training term

def read_raw_crite_data():
    '''
    read the data
    :return: pd.DataFrame
    '''
    header = ['ts_click', 'ts_cv', 'int1', 'int2', 'int3', 'int4',
              'int5', 'int6', 'int7', 'int8', 'cat1', 'cat2', 'cat3',
              'cat4', 'cat5', 'cat6', 'cat7', 'cat8', 'cat9']


    # raw_data = pd.read_table("./data/data.txt", names=header, nrows=1000)
    # raw_data['ts_click'] /= 86400
    # raw_data['ts_cv'] /= 86400
    # raw_data = pd.read_table("./criteo_conversion_logs/data_top5_campaign/7227c706.txt", names=header, index_col=False)ad3508b1 02a56acd
    raw_data = pd.read_table("./criteo_conversion_logs//all_data_top_campaign/93ec533b.txt", names=header, index_col=False)
    return raw_data

def fill_nan(raw_data):
    header = ['ts_click', 'ts_cv', 'int1', 'int2', 'int3', 'int4',
              'int5', 'int6', 'int7', 'int8', 'cat1', 'cat2', 'cat3',
              'cat4', 'cat5', 'cat6', 'cat7', 'cat8', 'cat9']

    # int1 -> -1
    raw_data['int1'] = raw_data['int1'].fillna(-1)

    # int2 -> 0
    raw_data['int2'] = raw_data['int2'].fillna(0.0)

    # int3 -> -1
    raw_data['int3'] = raw_data['int3'].fillna(-1)

    # int4 -> 0
    raw_data['int4'] = raw_data['int4'].fillna(0.0)

    # int 5 -> -1
    raw_data['int5'] = raw_data['int5'].fillna(-1)

    # int 6、7、8 -> 0
    raw_data['int6'] = raw_data['int6'].fillna(0.0)
    raw_data['int7'] = raw_data['int7'].fillna(0.0)
    raw_data['int8'] = raw_data['int8'].fillna(0.0)

    # raw_data.drop(['int1','int4'], axis=1, inplace=True)
    return raw_data

def get_dummies(raw_data):
    header = ['cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat7', 'cat8', 'cat9']
    one_hot_cat_data = pd.get_dummies(raw_data, prefix=header)
    shape = one_hot_cat_data.shape
    return one_hot_cat_data


def get_PCA(dummies):
    pca = PCA(n_components=50, copy=True)
    pca_data = pca.fit_transform(dummies)

    return pca_data

def get_standardize(pca):
    for i in range(pca.shape[1]):
        mean = np.mean(pca[:,i])
        std = np.std(pca[:,i])
        pca[:,i] = (pca[:,i] - mean) / std

    return pca

def check_cv_is_observed(ts_cv, ts_beginning_test_for_cvr):
    '''
    check if cv of the data is observed. If not, ts_cv is made NaN
    :param ts_cv: np.array
    :param ts_beginning_test_for_cvr: int
    '''
    ts_cv[ts_cv > ts_beginning_test_for_cvr] = np.nan

    return ts_cv


def drop_ts_cols(data):
    '''
    drop the timestamps columns
    :param data: pd.DataFrame
    :return: None
    '''
    data.drop(['ts_click', 'ts_cv'], axis=1, inplace=True)



label = "93ec533b"
data = read_raw_crite_data()

ts_click = data.loc[:, 'ts_click']
ts_cv = data.loc[:, 'ts_cv']
ts_click = np.array(ts_click)
ts_cv = np.array(ts_cv)

data = fill_nan(data)
data = get_dummies(data)
drop_ts_cols(data)
data = get_PCA(data)
data = get_standardize(data)

np.savetxt("./data_top_pca/" + label +".txt", data, fmt='%f', delimiter=',')

print("Finish Reading Data")