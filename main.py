# coding: utf-8
import argparse
import time
import math
import os, glob, sys
import numpy as np
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation, Dropout, LSTM, GRU, RNN, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

parser = argparse.ArgumentParser(description='Keras model for series prediction.')
parser.add_argument('--data_path', type=str, default='./',
                    help='location of the data')
parser.add_argument('--folder_path', type=str, default='./data',
                    help='location of the work folder')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, GRU)')
parser.add_argument('--activation', type=str, default='tanh',
                    help='type of activation function (linear, relu, sigmoid, tanh)')
parser.add_argument('--bidirectional', type=bool, default=False,
                    help='use bidirectional recurrent net')
# parser.add_argument('--input_size', type=int, default=7,
#                     help='size of LSTM input')
parser.add_argument('--hidden_size', type=int, default=32,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--loss', type=str, default='mae',
                    help='type of loss function (mae, mse)')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.0005,
                    help='gradient clipping')
parser.add_argument('--patience', type=int, default=50,
                    help='upper epoch limit')
parser.add_argument('--epochs', type=int, default=500,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--time_step', type=int, default=32,
                    help='input time step')
parser.add_argument('--predict_len', type=int, default=3,
                    help='time gap between input and output')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--save_path', type=str, default='./results/',
                    help='path to save the final model')
args = parser.parse_args()
    

def print_args(args, save_path):
    print('data_path %s'%(args.data_path))
    print('model %s'%(args.model))
    print('activation %s'%(args.activation))
    print('loss %s'%(args.loss))
    print('bidirectional %s'%(args.bidirectional))
    # print('input_size %d'%(args.input_size))
    print('hidden_size %d'%(args.hidden_size))
    print('num_layers %d'%(args.nlayers))
    print('init_lr %.4f'%(args.lr))
    print('clip %.1f'%(args.clip))
    print('patience %d'%(args.patience))
    print('epochs %d'%(args.epochs))
    print('batch_size %d'%(args.batch_size))
    print('save_path %s'%(save_path))

###############################################################################
# Load data
###############################################################################

def read_data(xfile, yfile, iftrain=True):

    data = np.load(xfile) 
    print('finish reading %s'%(xfile))

    y = np.load(yfile)
    print('finish reading %s'%(yfile))

    print(data.shape)
    print(y.shape)

    return data, y

def main():
    global args
    train_X, train_y = read_data(args.data_path + 'train_data.npy', args.data_path + 'train_target.npy')
    val_X, val_y = read_data(args.data_path + 'dev_data.npy', args.data_path + 'dev_target.npy', False)
    test_X, test_y = read_data(args.data_path + 'test_data.npy', args.data_path + 'test_target.npy', False)
    
    input_size = (train_X.shape[1], train_X.shape[2])
    output_size = train_y.shape[1]
    if args.model in ["LSTM", "GRU"]:
        rnn_type = args.model
    else:
        raise ValueError
    
    if rnn_type == "LSTM":
        RNNlayer = LSTM
    elif rnn_type == "GRU":
        RNNlayer = GRU

    model = Sequential()
    if args.bidirectional:
        forward_layer = RNNlayer(args.hidden_size, activation=args.activation, return_sequences=True)
        backward_layer = RNNlayer(args.hidden_size, activation='relu', return_sequences=True, go_backwards=True)
        # forward_layer = RNNlayer(args.hidden_size, activation=args.activation)
        # backward_layer = RNNlayer(args.hidden_size, activation='relu', go_backwards=True)
        model.add(Bidirectional(forward_layer, backward_layer=backward_layer, input_shape=input_size))
        for i in range(args.nlayers-1):
            model.add(Bidirectional(forward_layer, backward_layer=backward_layer))
    else:
        model.add(RNNlayer(args.hidden_size, activation=args.activation, input_shape=input_size))
        for i in range(args.nlayers-1):
            model.add(RNNlayer(args.hidden_size, activation=args.activation))
    model.add(Flatten())
    model.add(Dense(16))
    model.add(Dropout(0.5))
    model.add(Dense(output_size))
    model.compile(loss=args.loss, optimizer=Adam(lr=args.lr, decay=args.clip))
    model.summary()
    # fit network
    early_stopping = EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True)
    history = model.fit(train_X, train_y, epochs=args.epochs, batch_size=args.batch_size, \
        callbacks=[early_stopping,],\
        validation_data=(val_X, val_y), verbose=1, shuffle=False)
    test_results = model.predict(test_X)
    print('Test MAE: %.4f' % mean_absolute_error(test_y, test_results))
    print('Test R^2: %.4f' % r2_score(test_y, test_results))

    if args.bidirectional:
        save_path = os.path.join(args.data_path, f'K_Bi_{args.model}')
    else:
        save_path = os.path.join(args.data_path, f'K_{args.model}')

    try:
        os.mkdir(save_path)
    except:
        pass

    print_args(args, save_path)

    model.save(save_path)
    np.save(os.path.join(save_path, 'best_result.npy'), test_results)
    # plot history
    fig, ax = pyplot.subplots(1, 1)
    ax.plot(history.history['loss'], label='train')
    ax.plot(history.history['val_loss'], label='val')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epochs')
    ax.legend()
    fig.savefig(os.path.join(save_path, 'history.png'), dpi=400, bbox_inches='tight')
    return 0

# from make_data import *

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

# try:
#     os.mkdir(os.path.join(args.data_path, 'temp/'))
# except:
#     pass
try:
    os.mkdir('./results/')
except:
    pass

# for dataset in glob.glob(os.path.join(args.data_path, '*.csv')):
#     d0 = read_csv(dataset,  parse_dates = ['date'], index_col=0, date_parser=parse)
#     d0.dropna(subset=d0.columns.difference(['air_temp', 'sea_temp']), inplace=True)
#     d0.fillna(-1, inplace=True)
#     d0.sort_values('date', inplace=True)
#     d0.to_csv(os.path.join(args.data_path, 'temp/', os.path.basename(dataset)))
# del d0
features = [0,1,2,3,5,6]
target = [5]
time_period = [('2015-1-1', '2019-5-31'), ('2019-6-1', '2019-7-1')]
interval = args.predict_len-1
feature_period = args.time_step
for dataset in glob.glob(os.path.join(args.data_path, '*.csv')):
    fname = os.path.splitext(os.path.basename(dataset))[0]
    print(fname)
    save_path = os.path.join(args.folder_path, f'{fname}/save_fp_{feature_period}_i_{interval+1}/')
    prepareData(dataset, time_period, save_path, features, target, feature_period, interval)
        
for data in glob.glob(os.path.join(args.folder_path, '*/*/')):
    args.data_path = data
    main()

# from view_results import *
#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os, sys, glob, pickle
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sns.set(style="ticks", font_scale=1.2)
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

def svm(X_val, y_val, X_train, y_train, X_test, search=0):
    if search:
        kernel = ['linear', 'rbf', 'poly']
        c = 2**np.arange(-5, 15, 2, dtype=float)
        gamma = 2**np.arange(3, -15, -2, dtype=float)
        epsilon = [0.01, 0.1, 1]
        shrinking = [True, False]
        degree = [3, 4, 5]

        svm_grid = {'kernel':kernel, 'C': c, 'gamma' : gamma, 'degree': degree}

        svm = SVR()
        svm_search = RandomizedSearchCV(svm, svm_grid, scoring='neg_mean_squared_error', cv=5, return_train_score=True, n_jobs=-1, n_iter=30, verbose=1)
        svm_search.fit(X_val, y_val)

        svm_model = SVR(**svm_search.best_params_)
    else:
        # svm_model = SVR(C=1, epsilon=0.01, gamma=0.1, kernel='linear')
        svm_model = SVR(C=1, epsilon=0.01, gamma=0.1, kernel='poly')
    
    svm_model.fit(X_train, y_train)
    svm_test_pred = svm_model.predict(X_test)
    return svm_test_pred

def rf(X_val, y_val, X_train, y_train, X_test, search=0):
    if search:
        ensemble_grid =  {'n_estimators': [(i+1)*5 for i in range(20)],
                         'criterion': ['mse', 'mae'],
                         'bootstrap': [True, False],
                         }

        ensemble = RandomForestRegressor()
        ensemble_search = RandomizedSearchCV(ensemble, ensemble_grid, scoring='neg_mean_squared_error', cv=5, return_train_score=True, n_jobs=-1, n_iter=10, verbose=1)
        ensemble_search.fit(X_val, y_val)
        
        rf_model = RandomForestRegressor(**ensemble_search.best_params_)
    else:
        rf_model = RandomForestRegressor(n_estimators=4)
        
    rf_model.fit(X_train, y_train)
    rf_test_pred = rf_model.predict(X_test)
    return rf_test_pred

def save(data_source, scaler, y_test, **preds):
    save_dir = data_source
    try:
        os.makedirs(save_dir)
    except:
        pass

    results = {}
    raw_results = {}
    raw_results['Ground_truth'] = y_test
    points = y_test
    
    results['MODEL'] = []
    results['MAE'] = []
    results['MSE'] = []
    results['RMSE'] = []
    results['R^2'] = []

    for key in preds:
        raw_results[key] = preds[key]
        results['MODEL'].append(key)
        results['MAE'].append(mean_absolute_error(preds[key], y_test))
        results['MSE'].append(mean_squared_error(preds[key], y_test))
        results['RMSE'].append(np.sqrt(mean_squared_error(preds[key], y_test)))
        results['R^2'].append(r2_score(y_test, preds[key]))
        points = np.vstack((points, preds[key]))

    pd.DataFrame(results).to_csv(os.path.join(save_dir, 'results.csv'), index=False)
    raw_results = pd.DataFrame(raw_results)
    raw_results.to_csv(os.path.join(save_dir, 'raw_results.csv'), index=False)
    
    ax_num = len(results['MODEL'])
    fig, ax = plt.subplots(ax_num, 1)
    fig.set_size_inches(12, 5.5*ax_num+2.4)
    if ax_num==1:
        ax = [ax,]
        
    target_index = 5
    try:
        mean, std = scaler.mean_[target_index], scaler.var_[target_index]
        print("Standrad revert")
        convert_results = raw_results*std+mean
    except:
        print("Minmax revert")
        dmax, dmin = scaler.data_max_[target_index], scaler.data_min_[target_index]
        convert_results = raw_results*(dmax-dmin)+dmin
    convert_results.to_csv(os.path.join(save_dir, 'convert_results.csv'), index=False)

    for i, col in enumerate(convert_results.columns[1:]):
        sns.lineplot(x=convert_results.index, y=convert_results[col], color="r", ax=ax[i])
        sns.lineplot(x=convert_results.index, y=convert_results['Ground_truth'], color="b", ax=ax[i])
        ax[i].legend([convert_results.columns[i+1], 'Observed Data'])
        ax[i].set_ylabel('Wave Height (m)')
        ax[i].set_xlabel('Data')
        ax[i].set_title('$R^2=%.4f$'%(results['R^2'][i]))
        
    fig.savefig(os.path.join(save_dir, 'raw.png'), dpi=400, bbox_inches='tight') 
    
    fig, ax = plt.subplots(1, ax_num)
    fig.set_size_inches(5.5*ax_num+1.8, 5.5)
    if ax_num==1:
        ax = [ax,]
    
    for i, col in enumerate(convert_results.columns[1:]):
        sns.regplot(x=convert_results[col], y=convert_results['Ground_truth'], color="b", ax=ax[i])
        # points_range = [convert_results['Ground_truth']-0.1, convert_results['Ground_truth']+0.1]
        # sns.regplot(x=points_range, y=points_range, color="r", ci=None, ax=ax[i])
        ax[i].legend([convert_results.columns[i+1]])
        ax[i].set_xlabel('Predicted Wave Height (m)')
        if i:
            ax[i].set_ylabel('')
        else:
            ax[i].set_ylabel('Observed Wave Height (m)')
        ax[i].set_title('$R^2=%.4f$'%(results['R^2'][i]))

    fig.savefig(os.path.join(save_dir, 'scatter.png'), dpi=400, bbox_inches='tight')
    plt.show()
    plt.close('all')
    
def praseResults(path):
    with open(os.path.join(path, 'param.txt'), 'r') as f:
        lines = f.read().splitlines()

    station_id = ''.join([c for c in lines[0] if c.isupper()])
    fp = int(lines[4].split(':')[-1])
    interval = int(lines[6].split(':')[-1])+1
    d = pd.read_csv(os.path.join(path, 'K_results/results.csv'))
    d.columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R^2']
    sLength = len(d['Model'])
    d['Station'] = pd.Series(np.full(sLength, station_id), index=d.index)
    d['Feature_Period'] = pd.Series(np.full(sLength, fp), index=d.index)
    d['Interval'] = pd.Series(np.full(sLength, interval), index=d.index)
    d = d[['Station', 'Feature_Period', 'Interval', 'Model', 'MAE', 'MSE', 'RMSE', 'R^2']]
    return d

def main(data_source, save_path):
    print(f"Read from {data_source}")
    
    X_train_raw = np.load(os.path.join(data_source, 'train_data.npy'))
    y_train_raw = np.load(os.path.join(data_source, 'train_target.npy'))

    X_train = X_train_raw.reshape(X_train_raw.shape[0], -1)
    y_train = y_train_raw.reshape(-1,)

    X_val_raw = np.load(os.path.join(data_source, 'dev_data.npy'))
    y_val_raw = np.load(os.path.join(data_source, 'dev_target.npy'))

    X_val = X_val_raw.reshape(X_val_raw.shape[0], -1)
    y_val = y_val_raw.reshape(-1,)

    X_test_raw = np.load(os.path.join(data_source, 'test_data.npy'))
    y_test_raw = np.load(os.path.join(data_source, 'test_target.npy'))

    X_test = X_test_raw.reshape(X_test_raw.shape[0], -1)
    y_test = y_test_raw.reshape(-1,)

    # lstm_test_pred = np.load(os.path.join(data_source, 'LSTM_model/best_result.npy')).reshape(-1,)
    preds = {}
    svm_test_pred = svm(X_val, y_val, X_train, y_train, X_test)
    preds['SVM'] = svm_test_pred
    rf_test_pred = rf(X_val, y_val, X_train, y_train, X_test)
    preds['RF'] = rf_test_pred
    try:
        preds['MLP'] = np.load(os.path.join(data_source, 'K_MLP/best_result.npy')).reshape(-1,)
    except:
        pass
    try:
        preds['Bi-LSTM'] = np.load(os.path.join(data_source, 'K_Bi_LSTM/best_result.npy')).reshape(-1,)
    except:
        pass
    try:
        preds['Bi-GRU'] = np.load(os.path.join(data_source, 'K_Bi_GRU/best_result.npy')).reshape(-1,)
    except:
        pass
    try:
        preds['LSTM'] = np.load(os.path.join(data_source, 'K_LSTM/best_result.npy')).reshape(-1,)
    except:
        pass
    try:
        preds['GRU'] = np.load(os.path.join(data_source, 'K_GRU/best_result.npy')).reshape(-1,)
    except:
        pass
    
    scaler = pickle.load(open(os.path.join(data_source, 'scaler.pkl'), 'rb'))
    save(os.path.join(save_path, data_source), scaler, y_test, **preds)

# ds = []
for data_source in glob.glob(os.path.join(args.folder_path, '*/*/')):
    main(data_source, args.save_path)
#     d = praseResults(data_source)
#     ds.append(d)
    
# ds = pd.concat(ds,axis=0)
# ds.to_csv(os.path.join(workforder, 'K_predictions.csv'), index=False)
# ds.to_csv(os.path.join(args.save_path, 'K_prediction_results.csv'), index=False)