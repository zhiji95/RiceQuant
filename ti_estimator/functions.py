from __future__ import print_function

import numpy as np
import math
import matplotlib.pyplot as plt
import time
import pandas as pd
import talib
import tensorflow as tf
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score,make_scorer
from rqdatac import *
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def get_dataframe(index_component, input_window_length, forecast_horizon, start_date, end_date,frequency, classnum, method):
#get dataframe from sigle company
#input shortcuts
  iwl = input_window_length
  fh = forecast_horizon
  method_list = ['ANN','SVM']
  if not (method in method_list):
    raise Exception(method + ' is not in' + str(method_list))
  ic = index_component
  this_ti = get_price(ic, start_date, end_date, frequency = frequency)
  dates = this_ti.index[:-fh]
  close = np.array(this_ti['close'])
  next_close = get_next_close(close, fh)
  high = np.array(this_ti['high'])
  low = np.array(this_ti['low'])
  Y = labeling(fh, close[:-fh], next_close, classes = classnum)
  sma = talib.SMA(close,iwl)[:-fh]
  ema = talib.EMA(close,iwl)[:-fh]
  atr = talib.ATR(high, low, close, iwl)[:-fh]
  admi = talib.ADX(high, low, close, iwl)[:-fh]
  cci = talib.CCI(high, low, close, iwl)[:-fh]
  roc = talib.ROC(close, timeperiod=iwl)[:-fh]
  rsi = talib.RSI(close, timeperiod=iwl)[:-fh]
  williams = talib.WILLR(high, low, close, timeperiod=iwl)[:-fh]
  slowk = stochasticK(high, low, close, iwl)[:-fh]
  slowd = talib.EMA(slowk,3)
#form a initial data frame
  df = pd.DataFrame({
                  'close price t': close[:-fh],
                  'close price t+s': next_close,
#                   'Y':Y,
                  'SMA':sma,
                  'EMA':ema,
                  'ATR':atr,
                  'ADMI':admi,
                  'CCI':cci,
                  'ROC': roc,
                  'RSI':rsi,
                  'Williams %R':williams,
                  'Stochastic %K':slowk,
                  'Stochastic %D':slowd,
                 }, index=dates)
# append label according to the method
  if method == 'SVM':
    df['Y'] = Y
    df = df.dropna(axis=0,how='any')
    y  = df['Y'].values
  if method == 'ANN':
    one_hot_code = get_onehotcode(Y)
    ys = []
    for i in range(len(one_hot_code[0])):
      string = 'Y' + str(i+1)
      ys.append(string)
      df[string]  = one_hot_code[:,i]
    df = df.dropna(axis=0,how='any')
    y = df[ys].values
  x = df[['ADMI', 'ATR', 'CCI', 'EMA', 'ROC', 'RSI', 'SMA', 'Stochastic %D',
                 'Stochastic %K', 'Williams %R']].values
  return x, y

def stochasticK( high, low, close, n):
  result = []
  for i in range(len(high)):
    if i < n:
      result.append(np.nan)
    else:
      hh = max(high[i - n: i])
      ll = min(low[i - n: i])
      ct = close[i]
      if (hh != ll):
        k = 100* (ct - ll)/(hh - ll)
      else:
        k = 0
      result.append(k)
  return np.array(result)


def get_next_close(close, fh):
  next_close = []
  size = len(close)
  for i in range(size - fh):
#     if (i + fh < size):
    next_close.append(close[i + fh])
#     else:
#       next_close[i] = NaN
  return next_close


def labeling(fh, ct, cts, classes):
#fh -- forecast horizon
#ct -- closing price of a stock on day t
#cts -- closing price of a stock on day t+s
# return value:
# 0 stands for Up
# 1 stands for no move
# 2 stands for down
  label = []
  
  for i in range(len(ct)):
    value = (cts[i] - ct[i])/ct[i]
    if classes == 3:
      threshold = threshold_generator(fh)
      if (value > threshold):
        label.append(0)
      elif (abs(value) <= threshold):
        label.append(1)
      else:
        label.append(2)
    elif classes == 2:
      if value > 0:
        label.append(0)
      else:
        label.append(1)
    else:
      print("Classes can only be 2 or 3")
  return label

def threshold_generator(fh):
  if fh == 1:
    ts = 0.63
  elif fh == 3:
    ts = 1.15
  elif fh == 5:
    ts = 1.49
  elif fh == 7:
    ts = 1.79
  elif fh == 10:
    ts = 2.14
  elif fh == 15:
    ts = 2.65
  elif fh == 20:
    ts = 3.08
  elif fh == 25:
    ts = 3.48
  elif fh == 30:
    ts = 3.94
  else:
    print('Please search online for more information.')
  return ts/100

def accuracy_calculator(prediction, truth):
  n = len(prediction)
  true = 0
  for i in range(n):
    if prediction[i] == truth[i]:
      true += 1
  accuracy = true/n
  return accuracy

def get_stat_ANN(iwl, fh, from_index, to_index, names, bs, epochs, neuron_num, start_date, end_date,freq, split = 0.8, verbose = False):
  classnum = neuron_num[-1]
  idx = names[from_index: to_index]
  d_test, d_train = [], []
  a_train = []
  a_test = []
  for name in idx:
    X, y = get_dataframe(name, iwl, fh, start_date, end_date, freq, classnum = classnum, method = 'ANN')
    edge = np.int(split*len(y))
    train_X, test_X, train_y, test_y = X[:edge], X[edge:],y[:edge],y[edge:]
    scaler = preprocessing.StandardScaler().fit(train_X)
    X_train_nomalized = scaler.transform(train_X)
    X_test_nomalized = scaler.transform(test_X)


    layer_num = len(neuron_num)
    nx = neuron_num[0]
    ny = neuron_num[layer_num - 1]
    weights, biases = {}, {}
    #get nomalized train and test set


    x = tf.placeholder('float32',[None, nx])
    y = tf.placeholder('float32',[None, ny])
    layer_l = x
    for l in range(layer_num - 2):
      input_num = neuron_num[l]
      hidden_num = neuron_num[l + 1]
      weights['w' + str(l+1)] = tf.Variable(tf.random_normal([input_num, hidden_num]))
      biases['b' + str(l+1)] = tf.Variable(tf.random_normal([hidden_num]))
#       layer_l = tf.layers.dense(layer_l, hudden_num, activation = tf.nn.relu)
      layer_l = tf.nn.relu(tf.add(tf.matmul(layer_l, weights['w' + str(l+1)]), biases['b' + str(l+1)]))

    weights['out'] = tf.Variable(tf.random_normal([hidden_num, ny]))
    biases['out'] = tf.Variable(tf.random_normal([ny]))
    prediction = tf.nn.softmax(tf.matmul(layer_l, weights['out']) + biases['out'])

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    d_train.append(nn_distribution(train_y, classnum))
    d_test.append(nn_distribution(test_y, classnum))

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        losses, epoches = [],[]
        for epoch in range(epochs):
            epoch_loss=0
            i=0
            while i < len(X_train_nomalized):
                start = i
                end = i + bs
                batch_x = np.array(X_train_nomalized[start:end])
                batch_y = np.array(train_y[start:end])

                _,c = sess.run([optimizer,cost] , feed_dict = {x: batch_x , y : batch_y})
                if verbose:
                    epoch_loss+= c
                i+= bs
            if (verbose) :
                  if (epoch % 1000 == 0):
                    print("Epoch",epoch , 'completed out of ' ,epochs, ' loss: ', epoch_loss )
                  losses.append(epoch_loss)
                  epoches.append(epoch)
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        print(tf.argmax(prediction,1).eval({x:X_test_nomalized}))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))


        a_test.append(accuracy.eval({x:X_test_nomalized , y: test_y}))
        a_train.append(accuracy.eval({x:X_train_nomalized , y: train_y}))
  return pd.DataFrame({
  '训练分布':d_train,
  '测试分布':d_test,
  '训练精度ANN': a_train,
  '测试精度ANN': a_test
  },index = idx)


def get_stat_SVM(iwl, fh, from_index, to_index, tuned_parameters, names,start_date, end_date,freq,classnum, split = 0.8):
  idx = names[from_index: to_index]
  d_test, d_train = [], []
  a_train = []
  a_test =  []
  for name in idx:
    #get cleaned technical indicators
    X, y = get_dataframe(name, iwl, fh, start_date, end_date,freq, classnum = classnum, method = 'SVM')
    edge = np.int(split*len(y))
    X_train, X_test, y_train, y_test = X[:edge], X[edge:],y[:edge],y[edge:]
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_nomalized = scaler.transform(X_train)
    X_test_nomalized = scaler.transform(X_test)

    clf = GridSearchCV(SVC(), tuned_parameters, cv = 5,
                  scoring = 'accuracy')
    clf.fit(X_train_nomalized, y_train)
    prediction_train = clf.predict(X_train_nomalized)
    prediction_test = clf.predict(X_test_nomalized)
    print("2 classes prediction: ", prediction_train, prediction_test)
    a_train.append(accuracy_score(y_train, prediction_train))
    a_test.append(accuracy_score(y_test, prediction_test))
#     z_train, t_train = ZeroTwoFrequency(y_train)
#     z_test, t_test = ZeroTwoFrequency(y_test)

    d_train.append(svm_distribution(y_train, classnum))
    d_test.append(svm_distribution(y_test, classnum))

  return pd.DataFrame({
      '训练分布':d_train,
      '测试分布':d_test,
      '训练精度SVM': a_train,
      '测试精度SVM': a_test
    },index = idx)

def get_stat_integrated(iwl, fh, from_index, to_index,tuned_parameters, names, bs, epochs, neuron_num, start_date, end_date,freq, split = 0.8, verbose = False):
    idx = names[from_index: to_index]
    d_test, d_train = [], []
    a_train = []
    a_test =  []
    classnum = neuron_num[-1]
    for name in idx:
#SVM

        X, y = get_dataframe(name, iwl, fh, start_date, end_date,freq, classnum = classnum, method = 'SVM')
        edge = np.int(split*len(y))
        X_train, X_test, y_train, y_test = X[:edge], X[edge:],y[:edge],y[edge:]
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train_nomalized = scaler.transform(X_train)
        X_test_nomalized = scaler.transform(X_test)

        clf = GridSearchCV(SVC(), tuned_parameters, cv = 5,
                      scoring = 'accuracy')
        clf.fit(X_train_nomalized, y_train)
        pred_train_SVM = clf.predict(X_train_nomalized)
        pred_test_SVM = clf.predict(X_test_nomalized)
# ANN
        X, y = get_dataframe(name, iwl, fh, start_date, end_date, freq, classnum = classnum, method = 'ANN')
        edge = np.int(split*len(y))
        train_X, test_X, train_y, test_y = X[:edge], X[edge:],y[:edge],y[edge:]
        scaler = preprocessing.StandardScaler().fit(train_X)
        X_train_nomalized = scaler.transform(train_X)
        X_test_nomalized = scaler.transform(test_X)


        layer_num = len(neuron_num)
        nx = neuron_num[0]
        ny = neuron_num[layer_num - 1]
        weights, biases = {}, {}
        #get nomalized train and test set


        x = tf.placeholder('float32',[None, nx])
        y = tf.placeholder('float32',[None, ny])
        layer_l = x
        for l in range(layer_num - 2):
            input_num = neuron_num[l]
            hidden_num = neuron_num[l + 1]
            weights['h' + str(l+1)] = tf.Variable(tf.random_normal([input_num, hidden_num]))
            biases['b' + str(l+1)] = tf.Variable(tf.random_normal([hidden_num]))

            layer_l = tf.nn.relu(tf.add(tf.matmul(layer_l, weights['h' + str(l+1)]), biases['b' + str(l+1)]))

        weights['out'] = tf.Variable(tf.random_normal([hidden_num, ny]))
        biases['out'] = tf.Variable(tf.random_normal([ny]))
        prediction = tf.nn.softmax(tf.matmul(layer_l, weights['out']) + biases['out'])

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
        optimizer = tf.train.AdamOptimizer().minimize(cost)
        d_train.append(nn_distribution(train_y, classnum))
        d_test.append(nn_distribution(test_y, classnum))

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            losses, epoches = [],[]
            for epoch in range(epochs):
                epoch_loss=0
                i=0
                while i < len(X_train_nomalized):
                    start = i
                    end = i + bs
                    batch_x = np.array(X_train_nomalized[start:end])
                    batch_y = np.array(train_y[start:end])

                    _,c = sess.run([optimizer,cost] , feed_dict = {x: batch_x , y : batch_y})
                    if verbose:
                        epoch_loss+= c
                    i+= bs
                if (verbose) :
                    if (epoch % 1000 == 0):
                        print("Epoch",epoch , 'completed out of ' ,epochs, ' loss: ', epoch_loss )
                    losses.append(epoch_loss)
                    epoches.append(epoch)
            correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
            pred_train_ANN = tf.argmax(prediction,1).eval({x:X_train_nomalized , y: train_y})
            pred_test_ANN = tf.argmax(prediction,1).eval({x:X_test_nomalized , y: test_y})

#             accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            prediction_train = pred_integration(pred_train_ANN, pred_train_SVM)
            prediction_test = pred_integration(pred_test_ANN, pred_test_SVM)
            print(pred_test_SVM)
            print(pred_test_ANN)
            print(prediction_test )
        a_train.append(updown_accuracy(y_train, pred_train_SVM, pred_train_ANN))
        a_test.append(updown_accuracy(y_test, pred_test_SVM, pred_test_ANN))
    return pd.DataFrame({
      '训练分布':d_train,
      '测试分布':d_test,
      '训练精度': a_train,
      '测试精度': a_test
    },index = idx)

def updown_accuracy(label, p_ann,p_svm):
    length = 0
    correct = 0
    for i in range(len(label)):
        if p_ann[i] == p_svm[i]:
            length += 1
            if p_ann[i] == label[i]:
                correct +=1
    return correct/length


def pred_integration(p1, p2):
    result = []
    for i in range(len(p1)):
        if (p1[i] * 2) == p2[i]:
            result.append(p1[i])
        else:
            result.append(1)
    return result


def svm_distribution(label, classnum):
  if (classnum == 2):
    z, t = ZeroTwoFrequency(label)
    return 'Up：' + str(z) + ' Down：' + str(t)
  if (classnum == 3):
    z, o, t = ZeroOneTwoFrequency(label)
    return 'Up：' + str(z) + ' Flat: ' + str(o) + ' Down：' + str(t)

def ZeroOneTwoFrequency(label):
  z, o, t = 0, 0, 0
  n = len(label)
  for i in label:
    if (i == 0):
      z += 1
    elif (i == 1):
      o += 1
    elif (i == 2):
      t += 1
    else:
      print("Illegal input")
  return round(z/n,2),round(o/n,2),round(t/n,2)

def save_model_ANN(iwl, fh, name, bs, epochs, neuron_num, start_date, end_date,freq, verbose = False):
  classnum = neuron_num[-1]
  X, Y = get_dataframe(name, iwl, fh, start_date, end_date, freq, classnum = classnum, method = 'ANN')
  scaler = preprocessing.StandardScaler().fit(X)
  X_nomalized = scaler.transform(X)

  layer_num = len(neuron_num)
  nx = neuron_num[0]
  ny = neuron_num[layer_num - 1]
  weights, biases = {}, {}
  #get nomalized train and test set


  x = tf.placeholder('float32',[None, nx])
  y = tf.placeholder('float32',[None, ny])
  layer_l = x
  for l in range(layer_num - 2):
    input_num = neuron_num[l]
    hidden_num = neuron_num[l + 1]
    weights['h' + str(l+1)] = tf.Variable(tf.random_normal([input_num, hidden_num]))
    biases['b' + str(l+1)] = tf.Variable(tf.random_normal([hidden_num]))

    layer_l = tf.nn.relu(tf.add(tf.matmul(layer_l, weights['h' + str(l+1)]), biases['b' + str(l+1)]))

  weights['out'] = tf.Variable(tf.random_normal([hidden_num, ny]))
  biases['out'] = tf.Variable(tf.random_normal([ny]))
  prediction = tf.nn.softmax(tf.matmul(layer_l, weights['out']) + biases['out'])
  tf.add_to_collection('pred', y)

  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
  optimizer = tf.train.AdamOptimizer().minimize(cost)
  saver = tf.train.Saver()

  with tf.Session() as sess:
      sess.run(tf.initialize_all_variables())
      losses, epoches = [],[]
      for epoch in range(epochs):
          epoch_loss=0
          i=0
          while i < len(X_nomalized):
              start = i
              end = i + bs
              batch_x = np.array(X_nomalized[start:end])
              batch_y = np.array(Y[start:end])

              _,c = sess.run([optimizer,cost] , feed_dict = {x: batch_x , y : batch_y})
              if verbose:
                  epoch_loss+= c
              i+= bs
          if (verbose) :
                if (epoch % 1000 == 0):
                  print("Epoch",epoch , 'completed out of ' ,epochs, ' loss: ', epoch_loss )
                losses.append(epoch_loss)
                epoches.append(epoch)
      path = "./"
      model_name = "ann" + str(classnum) +".ckpt"
      print("model name is : " + model_name)
      saver.save(sess, path + model_name)

def save_model_rf(ic, iwl, fh, start_date, end_date, freq, classnum, name):
    acc_scorer = make_scorer(accuracy_score)
    X, y = get_dataframe(ic, iwl, fh, start_date, end_date,frequency = freq,  classnum = classnum, method = 'SVM')
    scaler = preprocessing.StandardScaler().fit(X)
    X_normalized = scaler.transform(X)

    clf = GridSearchCV(estimator = RandomForestClassifier(min_samples_split=100,
                                  min_samples_leaf=20,max_depth=8,max_features='sqrt' ,random_state=10), 
                       param_grid = tuned_parameters, scoring=acc_scorer,cv=5)
    clf.fit(X_normalized, y/2)   
    with open('models/'+ name +str(classnum)+ '.pickle', 'wb') as fw:
        pickle.dump(clf, fw)  
def save_model_nb(ic, iwl, fh, start_date, end_date,tuned_parameters, freq,classnum, name):

    X, y = get_dataframe(ic, iwl, fh, start_date, end_date,frequency = freq, classnum = classnum, method = 'SVM')
    scaler = preprocessing.StandardScaler().fit(X)
    X_normalized = scaler.transform(X)
#     print("X_n: ",X_nomalized)
    clf = GaussianNB()
    clf.fit(X_normalized, y)   
    with open('models/'+ name +str(classnum)+ '.pickle', 'wb') as fw:
        pickle.dump(clf, fw)
        
        
def save_model_SVM(ic, iwl, fh, start_date, end_date, tuned_parameters, freq,classnum, name):
    X, y = get_dataframe(ic, iwl, fh, start_date, end_date,frequency = freq, classnum = classnum, method = 'SVM')
    scaler = preprocessing.StandardScaler().fit(X)
    X_normalized = scaler.transform(X)
    print("X_n: ",X_normalized)
    clf2 = GridSearchCV(SVC(), tuned_parameters, cv=5,
                        scoring = 'accuracy')
    clf2.fit(X_normalized, y)
    with open('models/'+ name + str(classnum)+'.pickle', 'wb') as fw:
        pickle.dump(clf2, fw)
def get_stat_nb(iwl, fh, from_index, to_index, names,start_date, end_date,freq,classnum, split = 0.8):
  idx = names[from_index: to_index]
  a_train = []
  a_test =  []
  for name in idx:
    #get cleaned technical indicators
    X, y = get_dataframe(name, iwl, fh, start_date, end_date,freq, classnum = classnum, method = 'SVM')
    edge = np.int(split*len(y))
    X_train, X_test, y_train, y_test = X[:edge], X[edge:],y[:edge],y[edge:]
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_nomalized = scaler.transform(X_train)
    X_test_nomalized = scaler.transform(X_test)

    clf = GaussianNB()
    clf.fit(X_train_nomalized, y_train)
    prediction_train = clf.predict(X_train_nomalized)
    prediction_test = clf.predict(X_test_nomalized)
    print("2 classes prediction: ", prediction_train, prediction_test)
    a_train.append(accuracy_score(y_train, prediction_train))
    a_test.append(accuracy_score(y_test, prediction_test))
#     z_train, t_train = ZeroTwoFrequency(y_train)
#     z_test, t_test = ZeroTwoFrequency(y_test)

  return pd.DataFrame({
      '训练精度nb': a_train,
      '测试精度nb': a_test
    },index = idx)

def get_stat_rf(iwl, fh, from_index, to_index, tuned_parameters, names,start_date, end_date,freq,classnum, split = 0.8):
  idx = names[from_index: to_index]
  a_train = []
  a_test =  []
  acc_scorer = make_scorer(accuracy_score)
  for name in idx:
    #get cleaned technical indicators
    X, y = get_dataframe(name, iwl, fh, start_date, end_date,freq, classnum = classnum, method = 'SVM')
    edge = np.int(split*len(y))
    X_train, X_test, y_train, y_test = X[:edge], X[edge:],y[:edge],y[edge:]
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_normalized = scaler.transform(X_train)
    X_test_normalized = scaler.transform(X_test)

    clf = GridSearchCV(estimator = RandomForestClassifier(min_samples_split=100,
                                  min_samples_leaf=20,max_depth=8,max_features='sqrt' ,random_state=10), 
                       param_grid = tuned_parameters, scoring=acc_scorer,cv=5)
    clf.fit(X_train_normalized, y_train/2)
    prediction_train = clf.predict(X_train_normalized)
    prediction_test = clf.predict(X_test_normalized)
    print("2 classes prediction: ", prediction_train, prediction_test)
    a_train.append(accuracy_score(y_train, prediction_train*2))
    a_test.append(accuracy_score(y_test, prediction_test*2))

  return pd.DataFrame({
      '训练精度rf': a_train,
      '测试精度rf': a_test
    },index = idx)

def save_dataframe(df, name, sheetname = 'two classes'):
    with pd.ExcelWriter(name + '.xls') as writer:
        df.to_excel(writer,sheet_name = sheetname)
        
        
def ZeroTwoFrequency(label):
  z, t = 0,0
  n = len(label)
  for i in label:
    if (i == 0):
      z += 1
    elif (i == 2):
      t += 1
    else:
      print("Illegal input")
  return round(z/n,2), round(t/n,2)

def ZeroOneTwoFrequency_nn(label):
  z, o, t = 0, 0, 0
  n = len(label)
  for i in label:
    if (i[0] == 1):
      z += 1
    elif (i[1] == 1):
      o += 1
    elif (i[2] == 1):
      t += 1
    else:
      print("Illegal input")
  return round(z/n,2),round(o/n,2),round(t/n,2)

def ZeroTwoFrequency_nn(label):
  z, t = 0,0
  n = len(label)
  for i in label:
    if (i[0] == 1):
      z += 1
    elif (i[1] == 1):
      t += 1
    else:
      print("Illegal input")
  return round(z/n,2), round(t/n,2)

def nn_distribution(label, classnum):
  if (classnum == 2):
    z, t = ZeroTwoFrequency_nn(label)
    return 'Up：' + str(z) + ' Down：' + str(t)
  if (classnum == 3):
    z, o, t = ZeroOneTwoFrequency_nn(label)
    return 'Up：' + str(z) + ' 平: ' + str(o) + ' Down：' + str(t)

def ann_parameter_selection(name, neuron_num, class_num, batch_size, hm_epochs, iwl, fh,start_date,end_date, verbose = False, frequency = '1d', split = 0.8):
  if (len(neuron_num) < 3):
    print ("len(neuron_num) should be greater than 2")
  #get dataframe
  X_raw ,Y_raw  = get_dataframe(name, iwl, fh, freq = frequency, classnum = class_num, method = 'ANN')
  layer_num = len(neuron_num)
  nx = neuron_num[0]
  ny = neuron_num[layer_num - 1]
  weights, biases = {}, {}
  #get nomalized train and test set
  edge = np.int(split * len(Y_raw))
  train_x, test_x, train_y, test_y = X_raw[:edge], X_raw[edge:], Y_raw[:edge], Y_raw[edge:]
  scaler = preprocessing.StandardScaler().fit(train_x)
  train_x = scaler.transform(train_x)
  test_x = scaler.transform(test_x)

  x = tf.placeholder('float32',[None, nx])
  y = tf.placeholder('float32',[None, ny])
  layer_l = x
  for l in range(layer_num - 2):
    input_num = neuron_num[l]
    hidden_num = neuron_num[l + 1]
    weights['h' + str(l+1)] = tf.Variable(tf.random_normal([input_num, hidden_num]))
    biases['b' + str(l+1)] = tf.Variable(tf.random_normal([hidden_num]))

    layer_l = tf.add(tf.matmul(layer_l, weights['h' + str(l+1)]), biases['b' + str(l+1)])

  weights['out'] = tf.Variable(tf.random_normal([hidden_num, ny]))
  biases['out'] = tf.Variable(tf.random_normal([ny]))
  prediction = tf.matmul(layer_l, weights['out']) + biases['out']

  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
  optimizer = tf.train.AdamOptimizer().minimize(cost)


  with tf.Session() as sess:
      sess.run(tf.initialize_all_variables())
      losses, epoches = [],[]
      for epoch in range(hm_epochs):
          epoch_loss=0
          i=0
          while i < len(train_x):
              start = i
              end = i + batch_size
              batch_x = np.array(train_x[start:end])
              batch_y = np.array(train_y[start:end])

              _,c = sess.run([optimizer,cost] , feed_dict = {x: batch_x , y : batch_y})
              if verbose:
                epoch_loss+= c
              i+= batch_size
          if (verbose) :
            if (epoch % 1000 == 0):
              print("Epoch",epoch , 'completed out of ' ,hm_epochs, ' loss: ', epoch_loss )
            losses.append(epoch_loss)
            epoches.append(epoch)
      correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
      print(tf.argmax(prediction,1).eval({x:test_x , y: test_y}))
      accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
      a_test = accuracy.eval({x:test_x , y: test_y})
      a_train = accuracy.eval({x:train_x , y: train_y})

  return a_test, a_train

def get_onehotcode(Y):
    array = np.array(Y)
    enc = preprocessing.OneHotEncoder()
    enc.fit(np.array_split(array, len(array)))
    one_hot_code = enc.transform(np.array_split(array, len(array))).toarray()
    return one_hot_code


class info():
  def __init__(self, name):
    tp = type(name)
    if (tp) != str:
      raise Exception("the input type must be str, but got ", tp)
    self.name = str(name)
    get_stat_SVM_dic = {
      'iw':'input window length',
      'fh':'forecast horizon',
      'from_index':'start index of names you would like to get stat',
      'to_index':'the end index of names you would like to get stat',
      'tuned_parameters':'the parameter of SVC, for example: gamma, C, kernel. Please save them as a list as the example:' +
"tuned_parameters = {'kernel': ['rbf'], 'gamma': gammas,  'C': Cs}",
      'names':'the list of names of futures or shares',
      'start_date':'the start date of the data',
      'end_date':'the end date of the date',
      'freq':'the frequency of data, for example "1d", "1m","1h"',
      'classnum':'the number of class you try to classify',
      'split':'the percentage of training set, default is 0.8',
      'verbose':'whether you would like to see some intermedium results, default is False'
    }

    get_stat_ANN_dic = {
      'iwl':'input window length',
      'fh':'forecast horizon',
      'from_index':'start index of names you would like to get stat',
      'to_index':'the end index of names you would like to get stat',
      'names':'the list of names of futures or shares',
      'bs':'batch size (a hyper parameter need to be tuned)',
      'epochs':'how many epochs you would like to run',
      'neuron_num':'a list represent a neural network, each value in the list means the number of neurons in each layer',
      'start_date':'the start date of the data',
      'end_date':'the end date of the date',
      'freq':'the frequency of data, for example "1d", "1m","1h"',
      'split':'the percentage of training set, default is 0.8',
      'verbose':'whether you would like to see some intermedium results, default is False'
    }
    self.function_dic_chinese = {
      'get_stat_ANN':'得到每一支期货的测试集，训练集分布及ANN算法的预测',
      'get_stat_SVM':'得到每一支期货的测试集，训练集分布及SVM算法的预测',
      'get_dataframe':'得到每一支期货技术指标及收盘价，并分成X和y',
      'ann_parameter_selection': '测试不同输入参数下ANN的精度'
    }
    self.parameter_dic = {
      'get_stat_ANN' : get_stat_ANN_dic,
      'get_stat_SVM' : get_stat_SVM_dic,
    }


  def function_intro(self):
    print(self.function_dic_chinese[self.name])
  def all_parameters(self):
    print("parameters of "+self.name+" are as follow")
    print(self.parameter_dic[self.name].keys())
    print('Please type "parameters_intro(parameter you try to know)"')
  def parameters_intro(self, parameter_name):
    print(self.parameter_dic[self.name][parameter_name])
  def print_all_parameter_intro(self):
    d = self.parameter_dic[self.name]
    for name in d.keys():
      print(name + " : " + d[name])

