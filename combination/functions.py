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
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score,make_scorer
from sklearn.ensemble import RandomForestClassifier

def rf():
    #return a random forest model
    param_test1 = {'n_estimators':[10, 30, 50, 70]}
    clf = GridSearchCV(estimator = RandomForestClassifier(min_samples_split=100,
                                  min_samples_leaf=20,max_depth=8,max_features='sqrt' ,random_state=10), 
                       param_grid = param_test1, scoring='roc_auc',cv=5)
    return clf
def ann():
#     return an artificial neural network model, the construction is 31, 31, 10, 5, 2, activate function are ReLU except the out layer(softmax)
    model = Sequential()
    model.add(Dense(31, activation = 'relu', input_shape = (31, )))
    model.add(Dropout(0.1))
    model.add(Dense(10, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation = 'relu'))
    model.add(Dense(2, activation = 'softmax', bias_regularizer=regularizers.l1(0.00001)))
    adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
    model.compile(loss='binary_crossentropy',
              optimizer = adadelta, metrics=['binary_accuracy'])
    return model

def panel_append(p):
#   define a function for each element from a panel to append other elements and form a list 
    result = []
    r = np.concatenate((np.r_[1:21],np.r_[40:241:20]))
    for m in r:
        rt = (p/p.shift(m)) - 1
        result.append(rt)
    return list(np.array(result).T)

def lstm():
#     return a lstm model
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
#   transform the y into a proper data type for the input of lstm
    return np.reshape(erd,(erd.shape[0], erd.shape[1], 1))
def lstm_append(p):
    result = []
    r = np.r_[1:241]
    for m in r:
        rt = (p/p.shift(m)) - 1
        result.append(rt)
    return list(np.array(result).T)
def get_return(model, start_date = '20050104', end_date ='20180709', index_name = '399001.XSHE',cost = 0, k = 5):
    valid_inputs =  ['lstm','rf','log','ann']
    if not model in valid_inputs:
        print('illegal input, inputs should be in', valid_inputs)
        return
    if (type(model) != str):
        print('wrong input type, model should be string.')
    else:
        if model == 'lstm':
            returns, dates = get_return_lstm(start_date, end_date, index_name, cost, k)
        if model == 'rf':
            returns, dates = get_return_rf(start_date, end_date, index_name, cost, k)
        if model == 'log':
            returns, dates = get_return_log(start_date, end_date, index_name, cost, k)
        if model == 'ann':
            returns, dates = get_return_ann(start_date, end_date, index_name, cost, k)
    return returns, dates
def get_return_lstm(start_date, end_date, index_name = '399001.XSHE', cost = 0,k = 5):
    index = []
    returns, dates = [],[]
    ics = index_components(index_name)
    ps = get_price(ics, start_date, end_date,frequency = '1d')['close']
    
    for i in np.r_[0:len(ps.index)-1000:250]:
# Get close prices
        p = ps[i:i+1000].dropna(axis = 'columns', how = 'any')
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
        model_lstm.fit(sandlize(np.array(X_train)), y_train_oh,epochs = 80,callbacks=[early_stopping])
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
            returns.append(this_profit/k)
            dates.append(ps.index[i+t])
    return returns, dates


def get_return_rf(start_date, end_date, index_name = '399001.XSHE', cost = 0,k = 5):
    profit_all, profit_everyday, accuracy_train = [],[],[]
    index = []
    returns,dates = [],[]
    ics = index_components(index_name)
    ps = get_price(ics, start_date, end_date,frequency = '1d')['close']
    
    for i in np.r_[0:len(ps.index)-1000:250]:
        print(i)
# Get close prices
        p = ps[i:i+1000].dropna(axis = 'columns', how = 'any')
        ics_new = p.columns
        if (len(ics_new) < k*2):
            print(len(ics_new))
            continue
        rt = (p - p.shift(1))/p
        med = rt.median(axis = 1)
        x_all = p.apply(lambda z : panel_append(z))
        X_train = []
        y_train = []
        for ic in ics_new:
            x = x_all[ic]
            y = (rt[ic] > med) * 1
            X_train += list(x)[241:749]
            y_train += list(y)[242:750]
        if (len(X_train) != len(y_train)):
            print('i:',i)
            continue
        model_rf = rf()
        model_rf.fit(X_train, y_train)
        pred_train = model_rf.predict(X_train)
        acc_train = accuracy_score(y_train, pred_train)
        dic = pd.DataFrame()
        for ic in ics_new:
            y = (rt[ic] > med) * 1
            X_test = list(x_all[ic])[750: -1]   
            proba_test = model_rf.predict_proba(X_test)
            dic[ic] = np.array(proba_test)[:,1]
        if (len(dic.index) < 248):
            continue
        for t in np.r_[750:999]:
            top_ks = dic.sort_values(by = dic.index[t-750], axis = 'columns', ascending = False).columns[:k]
            this_profit = 0
            for ic in top_ks:
                this_profit += rt[ic][t+1]
            returns.append(this_profit/k)
            dates.append(ps.index[i+t])
            
    return returns, dates

def get_return_log(start_date, end_date, index_name = '399001.XSHE', cost = 0,k = 5):
    profit_all, profit_everyday, accuracy_train = [],[],[]
    index = []
    returns,dates = [],[]
    ics = index_components(index_name)
    ps = get_price(ics, start_date, end_date,frequency = '1d')['close']
    
    for i in np.r_[0:len(ps.index)-1000:250]:
# Get close prices
        p = ps[i:i+1000].dropna(axis = 'columns', how = 'any')
        ics_new = p.columns
        if (len(ics_new) < k*2):
            print(len(ics_new))
            continue
        rt = (p - p.shift(1))/p
        med = rt.median(axis = 1)
        x_all = p.apply(lambda z : panel_append(z))
        X_train = []
        y_train = []
        for ic in ics_new:
            x = x_all[ic]
            y = (rt[ic] > med) * 1
            X_train += list(x)[241:749]
            y_train += list(y)[242:750]
        if (len(X_train) != len(y_train)):
            print('i:',i)
            continue
        model_log = LogisticRegression()
        model_log.fit(X_train, y_train)
        pred_train = model_log.predict(X_train)
        acc_train = accuracy_score(y_train, pred_train)
        dic = pd.DataFrame()
        for ic in ics_new:
            y = (rt[ic] > med) * 1
            X_test = list(x_all[ic])[750: -1]   
            proba_test = model_log.predict_proba(X_test)
            dic[ic] = np.array(proba_test)[:,1]
        if (len(dic.index) < 248):
            continue
        for t in np.r_[750:999]:
            top_ks = dic.sort_values(by = dic.index[t-750], axis = 'columns', ascending = False).columns[:k]
            this_profit = 0
            for ic in top_ks:
                this_profit += rt[ic][t+1]
#             print(this_profit/k, list(ps.index)[i+t])
            returns.append(this_profit/k)
            dates.append(ps.index[i+t])
            
    return returns, dates

def get_return_ann(start_date, end_date, index_name = '399001.XSHE' , cost = 0,k = 5):
    returns, dates= [],[]
    index = []
    ics = index_components(index_name)
    ps = get_price(ics, start_date, end_date,frequency = '1d')['close']
    
    for i in np.r_[0:len(ps.index)-1000:250]:
# Get close prices
        p = ps[i:i+1000].dropna(axis = 'columns', how = 'any')
        ics_new = p.columns
        print(len(ics_new))
        if (len(ics_new) < k*2):
            print(len(ics_new))
            continue
        rt = (p - p.shift(1))/p
        med = rt.median(axis = 1)
        x_all = p.apply(lambda z : panel_append(z))
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
        model_ann = ann()
        early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=0, mode='min')
        model_ann.fit(np.array(X_train), y_train_oh,epochs = 80, callbacks = [early_stopping])
        pred_train = model_ann.predict(np.array(X_train))
        acc_train = model_ann.evaluate(np.array(X_train), pred_train)
        dic = pd.DataFrame()
        for ic in ics_new:
            y = (rt[ic] > med) * 1
            X_test = list(x_all[ic])[750: -1]   
            proba_test = model_ann.predict(np.array(X_test))
            dic[ic] = np.array(proba_test)[:,1]
        if (len(dic.index) < 248):
            print(len(dic.index))
            continue
        for t in np.r_[750:999]:
        
            top_ks = dic.sort_values(by = dic.index[t-750], axis = 'columns', ascending = False).columns[:k]
            for ic in top_ks:
                this_profit = rt[ic][t+1]/k
            returns.append(this_profit)
            dates.append(ps.index[i+t])
    return returns, dates
def prices(returns, base):
    # Converts returns into prices
    s = [base]
    for i in range(len(returns)):
        s.append(base * (1 + returns[i]))
    return np.array(s)

def lpm(returns, threshold, order):
    # This method returns a lower partial moment of the returns
    # Create an array he same length as returns containing the minimum return threshold
    threshold_array = np.empty(len(returns))
    threshold_array.fill(threshold)
    # Calculate the difference between the threshold and the returns
    diff = threshold_array - returns
    # Set the minimum of each to 0
    diff = diff.clip(min=0)
    # Return the sum of the different to the power of order
    return np.sum(diff ** order) / len(returns)

def var(returns, alpha):
    # This method calculates the historical simulation var of the returns
    sorted_returns = np.sort(returns)
    # Calculate the index associated with alpha
    index = int(alpha * len(sorted_returns))
    # VaR should be positive
    return abs(sorted_returns[index])
def cvar(returns, alpha):
    # This method calculates the condition VaR of the returns
    sorted_returns = np.sort(returns)
    # Calculate the index associated with alpha
    index = int(alpha * len(sorted_returns))
    # Calculate the total VaR beyond alpha
    sum_var = sorted_returns[0]
    for i in range(1, index):
        sum_var += sorted_returns[i]
    # Return the average VaR
    # CVaR should be positive
    return abs(sum_var / index)
def sortino_ratio(er, returns, rf = 0, target=0):
    return (er - rf) / math.sqrt(lpm(returns, target, 2))
def dd(returns, tau):
    # Returns the draw-down given time period tau
    values = prices(returns, 100)
    pos = len(values) - 1
    pre = pos - tau
    drawdown = float('+inf')
    # Find the maximum drawdown given tau
    while pre >= 0:
        dd_i = (values[pos] / values[pre]) - 1
        if dd_i < drawdown:
            drawdown = dd_i
        pos, pre = pos - 1, pre - 1
    # Drawdown should be positive
    return abs(drawdown)
def max_dd(returns):
    # Returns the maximum draw-down for any tau in (0, T) where T is the length of the return series
    max_drawdown = float('-inf')
    for i in range(0, len(returns)):
        drawdown_i = dd(returns, i)
        if drawdown_i > max_drawdown:
            max_drawdown = drawdown_i
    # Max draw-down should be positive
    return abs(max_drawdown)
def get_stat_a(returns):
    df = pd.DataFrame(returns)
    m = np.average(returns)
    std = np.std(returns)
    ste = std/np.sqrt(len(returns))
    t_stat = m/ste
    median = np.median(returns)
    q1 = np.percentile(returns,25)
    q3 = np.percentile(returns,75)
    hit_ratio = len(np.array(returns)[np.array(returns) > 0] )/len(returns)
    maximum = max(returns)
    minimum = min(returns)
    skew = df.skew()[0]
    kurt = df.kurt()[0]
    return [m,ste,t_stat,minimum,q1,median,q3,maximum,hit_ratio,std,skew,kurt]
def get_stat_b(returns):
    from scipy.stats import norm
    VaR1 = var(returns, 0.01)
    VaR5 = var(returns, 0.05)
    CVaR1 = cvar(returns, 0.01)
    CVaR5 = cvar(returns, 0.05)
    return [VaR1, CVaR1, VaR5, CVaR5,max_dd(returns)]
def get_stat_c(returns):
    pa_returns, sharp_ratios = [], []
    sortino_ratios, stds, dstds, ers = [],[],[],[]
    i = 0
    
    while i < len(returns):
        annual_return = returns[i:i+250]
        dstd = np.std(np.array(annual_return)[np.array(annual_return) < 0])
        annual_profit = [i+1 for i in annual_return]
        pa_return = np.array(annual_profit).cumprod()[-1]
        m = np.average(annual_return)
        std = np.std(annual_return)
        sharp_ratio = (m*np.sqrt(250))/std
        sortino_ratio = (m*np.sqrt(250))/dstd
        sharp_ratios.append(sharp_ratio)
        sortino_ratios.append(sortino_ratio)
        stds.append(std)
        dstds.append(dstd)
        i+=250
    return [np.average(pa_returns),np.average(stds),np.average(dstds), np.average(sharp_ratios),np.average(sortino_ratios)]