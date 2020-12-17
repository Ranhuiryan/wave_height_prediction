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

from make_data import *

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

from view_results import *

# ds = []
for data_source in glob.glob(os.path.join(args.folder_path, '*/*/')):
    main(data_source, args.save_path)
#     d = praseResults(data_source)
#     ds.append(d)
    
# ds = pd.concat(ds,axis=0)
# ds.to_csv(os.path.join(workforder, 'K_predictions.csv'), index=False)
# ds.to_csv(os.path.join(args.save_path, 'K_prediction_results.csv'), index=False)