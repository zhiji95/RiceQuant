import rqdatac as rq
from rqdatac import *
rq.init()

import numpy as np
import keras
from keras.layers.core import Dense, Activation, Dropout,Reshape
from keras.models import Sequential
from keras.optimizers import SGD,Adam,Adadelta
from keras import regularizers
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.metrics import accuracy_score,make_scorer
from sklearn.ensemble import RandomForestClassifier
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping
import pandas as pd

def lstm():
    model = Sequential()
    model.add(LSTM(25, return_sequences = False,
                   input_shape=(240, 1)))  # returns a sequence of vectors of dimension 32
    model.add(Dropout(0.1))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['binary_accuracy'])
    return model
def sandlize(erd):
    return np.reshape(erd,(erd.shape[0], erd.shape[1], 1))
def lstm_append(p):
    result = []
    r = np.r_[1:241]
    for m in r:
        rt = (p/p.shift(m)) - 1
        result.append(rt)
    return list(np.array(result).T)
def get_return_lstm(start_index, end_index,index_num = '399001.XSHE', cost = 0,k = 5):
    index = []
    returns, dates = [],[]
    for i in np.r_[start_index:end_index:250]:
# Get close prices
        ics = index_components(index_num)
        p = get_price(ics, '20050104','20180709',frequency = '1d')['close'][i:i+1000].dropna(axis = 'columns', how = 'any')
        ics_new = p.columns
        if (len(ics_new) < k*2):
            print(len(ics_new))
            continue
        rt = (p - p.shift(1))/p
        med = rt.median(axis = 1)
        x_all = p.apply(lambda z : lstm_append(z))
        X_train = []
        y_train = []
        for ic in ics_new:
            x = x_all[ic]
            y = (rt[ic] > med) * 1
            X_train += list(x)[241:749]
            y_train += list(y)[242:750]
        y_train_oh = keras.utils.to_categorical(y_train, num_classes=2)
        if (len(X_train) != len(y_train)):
            print('i:',i)
            continue
        model_lstm = lstm()
        early_stopping = EarlyStopping(monitor='loss', patience=5, verbose=0, mode='min')
        model_lstm.fit(sandlize(np.array(X_train)), y_train_oh,epochs = 15,callbacks=[early_stopping])
        pred_train = model_lstm.predict(sandlize(np.array(X_train)))
        acc_train = model_lstm.evaluate(sandlize(np.array(X_train)), pred_train, verbose = 0)
        print()
        dic = pd.DataFrame()
        for ic in ics_new:

            y = (rt[ic] > med) * 1
            X_test = list(x_all[ic])[750: -1]   
            proba_test = model_lstm.predict_proba(sandlize(np.array(X_test)))
            dic[ic] = np.array(proba_test)[:,1]
        if (len(dic.index) < 248):
            print(len(dic.index))
            continue
        for t in np.r_[750:999]:
        
            top_ks = dic.sort_values(by = dic.index[t-750], axis = 'columns', ascending = False).columns[:k]
            for ic in top_ks:
                this_profit = rt[ic][t+1]
                profit += this_profit
            returns.append(this_profit/k)
            dates.append(ps.index[i+t])
    return returns, dates

def get_profit_lstm(start_index, end_index,index_num = '399001.XSHE', cost = 0,k = 5):
    profit_all, profit_everyday, accuracy_train = [],[],[]
    index = []
    stds, hit_ratios, sharp_ratios = [],[],[]
    ms = []
    for i in np.r_[start_index:end_index:250]:
# Get close prices
        ics = index_components(index_num)
        p = get_price(ics, '20050104','20180709',frequency = '1d')['close'][i:i+1000].dropna(axis = 'columns', how = 'any')
        ics_new = p.columns
        if (len(ics_new) < k*2):
            print(len(ics_new))
            continue
        rt = (p - p.shift(1))/p
        med = rt.median(axis = 1)
        x_all = p.apply(lambda z : lstm_append(z))
        X_train = []
        y_train = []
        for ic in ics_new:
            x = x_all[ic]
            y = (rt[ic] > med) * 1
            X_train += list(x)[241:749]
            y_train += list(y)[242:750]
        y_train_oh = keras.utils.to_categorical(y_train, num_classes=2)
        if (len(X_train) != len(y_train)):
            print('i:',i)
            continue
        model_lstm = lstm()
        early_stopping = EarlyStopping(monitor='loss', patience=5, verbose=0, mode='min')
        model_lstm.fit(sandlize(np.array(X_train)), y_train_oh,epochs = 15,callbacks=[early_stopping])
        pred_train = model_lstm.predict(sandlize(np.array(X_train)))
        acc_train = model_lstm.evaluate(sandlize(np.array(X_train)), pred_train, verbose = 0)
        print()
        dic = pd.DataFrame()
        for ic in ics_new:

            y = (rt[ic] > med) * 1
            X_test = list(x_all[ic])[750: -1]   
            proba_test = model_lstm.predict_proba(sandlize(np.array(X_test)))
            dic[ic] = np.array(proba_test)[:,1]
        profit = 0
        ed_profit =  []
        if (len(dic.index) < 248):
            print(len(dic.index))
            continue
        for t in np.r_[750:999]:
        
            top_ks = dic.sort_values(by = dic.index[t-750], axis = 'columns', ascending = False).columns[:k]
            for ic in top_ks:
                this_profit = rt[ic][t+1]
                profit += this_profit
            ed_profit.append(this_profit/k)
        print(ed_profit)
        profit_all.append(profit)
        hit_ratio = (len(np.array(ed_profit)[np.array(ed_profit) > 0] )/len(ed_profit))
        hit_ratios.append(hit_ratio)
        profit_everyday.append(ed_profit)
        accuracy_train.append(acc_train)
        index.append(i)
        std = np.std(ed_profit)
        m = np.average(ed_profit)
        ms.append(m)
        stds.append(std)
        sharp_ratios.append(m/std)
        print("train accuracy is: ", acc_train,"profit: ", profit, " hit ratio: ", hit_ratio, ' sharp ratio: ', m/std,' daily return: ', m)
# Construct a dataframe
    df = pd.DataFrame(index = index)
    df['profits'] = profit_all
    df['everyday profit'] = profit_everyday
    df['training accuracy'] = accuracy_train
    df['standard deviation'] = stds
    df['hit ratio'] = hit_ratios
    df['sharp ratio'] = sharp_ratios
    df['daily return'] = ms
    return df

returns, dates = get_return_lstm(0,3000)

