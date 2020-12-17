#!/usr/bin/env python
# coding: utf-8
import pandas as pd
from pandas import read_csv
from datetime import datetime
from matplotlib import pyplot
from scipy import stats
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os, glob, sys, pickle
# import seaborn as sns

# sns.set(style="ticks", font_scale=1.4)
# pyplot.rcParams['font.sans-serif'] = ['SimHei']
# pyplot.rcParams['axes.unicode_minus'] = False

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

def readOceanData(station='XMD'):
    def parse(x):
        year = '2019'
        return datetime.strptime(year+' '+x, '%Y %m %d %H')
    dataset = read_csv('./data/ocean/ocean_data_2019.csv',  parse_dates = [['Month', 'Date', 'Time']], index_col=0, date_parser=parse)
    
    dataset = dataset[dataset['Station'] == station]
    dataset.drop(['Station', 'Lat.', 'Long.', 'Visibility', 'During_Past_6_hours_Precipitation', 'Surge_Height', 'Surge_Period'], axis=1, inplace=True)

    dataset.replace('C', 0, inplace=True)
    for c in dataset.columns.drop(['Wind Direction', 'Air pressure']):
        invalid_values = np.unique(dataset[dataset[c] > 90][c].values)
        for value in invalid_values:
            dataset[c].replace(value, np.nan, inplace=True)
    invalid_values = np.unique(dataset[dataset['Air pressure'] > 2000]['Air pressure'].values)
    for value in invalid_values:
        dataset['Air pressure'].replace(value, np.nan, inplace=True)
    dataset.dropna()
    
    dataset.columns = ['air_temp', 'wind_dir', 'wind_spd', 'air_pre', 'sea_temp', 'height', 'period']
    dataset.index.name = 'date'
    dataset.sort_index(inplace=True)
    
    dataset.to_csv('./data/ocean/station_%s.csv'%(station))


def parse(x):
    return datetime.strptime(x, '%Y %m %d %H')

if __name__ == '__main__':
    for dataset in glob.glob('./data/ocean/ext/*.csv'):
        d0 = read_csv(dataset,  parse_dates = ['date'], index_col=0, date_parser=parse)
        d0.dropna(subset=d0.columns.difference(['air_temp', 'sea_temp']), inplace=True)
        d0.fillna(-1, inplace=True)
        d0.sort_values('date', inplace=True)
        d0.to_csv('./data/ocean/full/'+os.path.basename(dataset))


    features = [0,1,2,3,5,6]
    target = [3]
    time_period = [('2015-1-1', '2019-5-31'), ('2019-7-2', '2019-12-31'), ('2019-6-1', '2019-7-1')]
    interval = 0
    # feature_period = 6
    # for interval in [0, 5, 11]:
    for feature_period in [1, 4, 6, 12]:
        for dataset in glob.glob('./data/ocean/full/station_*.csv'):
            fname = os.path.splitext(os.path.basename(dataset))[0]
            print(fname)
            save_path = f'./data/period_group_by_station_full/{fname}/save_fp_{feature_period}_i_{interval+1}/'
            prepareData(dataset, time_period, save_path, features, target, feature_period, interval)
        
