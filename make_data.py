#!/usr/bin/env python
# coding: utf-8

import os, pickle
import numpy as np
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler

def parseData(origin_data, features_period, stride, targets, interval=0, targets_period=1, features=[]):
    data = []
    target = []
    window_size = features_period + targets_period + interval
    data_num = (len(origin_data) - window_size) / stride + 1

    for i in range(int(data_num)):
        full_sample = origin_data[i*stride:i*stride + window_size, :]
        if not len(features):
            t_data = full_sample[:features_period]
        else:
            t_data = full_sample[:features_period][:, features]
        t_target = full_sample[(features_period+interval):][:, targets].reshape(-1,)

        data.append(t_data)
        target.append(t_target)
    return np.asarray(data), np.asarray(target)

def prepareData(data_source, time_periods, save_path, features, targets, features_period, interval=0, targets_period=1, stride=3):
    X = []
    y = []
    
    dataset = read_csv(data_source, header=0, index_col=0, parse_dates=True)
    # deal with abnormal values
    droped_col = ['wind_dir', 'air_pre']
    dataset.replace('C', 0, inplace=True)
    dataset = dataset.where(dataset['wind_dir']<400).where(dataset['air_pre']<2000).dropna()
    for c in dataset.columns.drop(droped_col):
        invalid_values = np.unique(dataset[dataset[c] > 90][c].values)
        for value in invalid_values:
            dataset[c].replace(value, np.nan, inplace=True)
    dataset.dropna(inplace=True)

    values = dataset.values
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(values)
    # scaler = StandardScaler().fit(values)
    
    for before, after in time_periods[:-1]:
        p_values = dataset.truncate(before, after).values.astype('float32')
        scaled = scaler.transform(p_values)
        data, target = parseData(scaled, features_period, stride, targets, interval, targets_period, features)
        X.append(data)
        y.append(target)
    X = np.vstack(X)
    y = np.vstack(y)
    print( "data size:")
    print(X.shape, y.shape)
    
    before, after = time_periods[-1]
    p_values = dataset.truncate(before, after).values.astype('float32')
    scaled = scaler.transform(p_values)
    X_test, y_test = parseData(scaled, features_period, 1, targets, interval, targets_period, features)
    print( "test size:")
    print(X_test.shape, y_test.shape)

    try:
        os.makedirs(save_path)
    except:
        print('Save folder exits')
        
    full_length = len(y)
    X_train, y_train = X[:int(full_length*0.8)], y[:int(full_length*0.8)]
    X_val, y_val = X[int(full_length*0.8):], y[int(full_length*0.8):]
    print( "train size:")
    print( np.shape(X_train), np.shape(y_train))
    print( "val size:")
    print( np.shape(X_val), np.shape(y_val))
    print( "test size:")
    print( np.shape(X_test), np.shape(y_test))

    pickle.dump(scaler, open(os.path.join(save_path, 'scaler.pkl'), 'wb'))
    np.save(save_path + 'train_data.npy', X_train)
    np.save(save_path + 'train_target.npy', y_train)
    np.save(save_path + 'dev_data.npy', X_val)
    np.save(save_path + 'dev_target.npy', y_val)
    np.save(save_path + 'test_data.npy', X_test)
    np.save(save_path + 'test_target.npy', y_test)

    features = [dataset.columns[i] for i in features]
    targets = [dataset.columns[i] for i in targets]   
    with open(save_path + 'param.txt','w') as f:
        f.writelines(
        f'''data source:{data_source}
        stride:{stride}
        features:{features}
        targets:{targets}
        features period:{features_period}
        targets period:{targets_period}
        interval:{interval}
        data size:{(X.shape, y.shape)}
        ''')
