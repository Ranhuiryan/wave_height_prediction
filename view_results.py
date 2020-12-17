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
    svm_test_pred = svm(X_val, y_val, X_train[:500], y_train[:500], X_test)
    preds['SVM'] = svm_test_pred
    rf_test_pred = rf(X_val, y_val, X_train[:500], y_train[:500], X_test)
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

if __name__ == '__main__':
    data_path = './train_data'
    save_path = './results'
    ds = []
    for data_source in glob.glob(os.path.join(data_path, '*/*/')):
        main(data_source, save_path)
        d = praseResults(data_source)
        ds.append(d)

    ds = pd.concat(ds,axis=0)
    # ds.to_csv(os.path.join(workforder, 'K_predictions.csv'), index=False)
    ds.to_csv(os.path.join(save_path, 'K_predictions.csv'), index=False)