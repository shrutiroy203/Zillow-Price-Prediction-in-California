import pandas as pd
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import itertools as it
from statsmodels.tsa.holtwinters import ExponentialSmoothing


import warnings
from math import sqrt
import matplotlib.pyplot as plt


def plot_all(data, lags):
    fig = plt.figure(figsize=(15, 3))
    ax = fig.add_subplot(111)
    ax.plot(data)
    plt.show()
    fig, ax = plt.subplots(1,2, figsize=(15,4))
    plot_acf(data, ax=ax[0], lags = lags)
    plot_pacf(data, ax=ax[1], lags=lags)
    plt.show()

def bic_sarima(data, p_vals, d_vals, q_vals, P_vals, D_vals, Q_vals, m_vals): 
    best_score, best_order, best_seasonal_order = float("inf"), None, None
    
    my_dict = {'p_vals': p_vals, 'd_vals': d_vals, 'q_vals': q_vals,
               'P_vals': P_vals, 'D_vals': D_vals, 'Q_vals': Q_vals, 'm_vals':m_vals}
    
    all_comb = it.product(*(my_dict[name] for name in my_dict))
    for comb in list(all_comb):  
        order = comb[0:3]
        seasonal_order = comb[3:]
        try:
            res = sm.tsa.statespace.SARIMAX(data,order=order,seasonal_order=seasonal_order,
                                            enforce_stationarity=False,enforce_invertibility=False).fit()
            bic=res.bic
            if bic < best_score:
                best_score, best_order, best_seasonal_order=bic, order,seasonal_order
        except:
            continue
    print((best_score, best_order, best_seasonal_order))
    
def diff(data, interval=1):
    diff = list()
    for i in range(interval, len(data)):
        value = data[i] - data[i - interval]
        diff.append(value)
    return diff

def adf_plot_detrend(data, interval, plot_detrend=True):    
    for i in range(interval):
        data = diff(data, interval=1)
    print(f'\n Detrended by {interval}')
    dftest = adfuller(data)
    dfoutput = pd.Series(dftest[0:2], index=['Test Statistic','p-value'])
    print (dfoutput)
    if plot_detrend:
        fig, ax = plt.subplots(1,1,figsize=(20,4))
        ax.plot(data)
        plt.tight_layout()
    return data
    
def evaluate_arima_model(X, arima_order, h):
    train_size = int(len(X) * 0.67)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    predictions = list()
    for t in range(len(test)-h):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit()
        yhat = np.array([model_fit.forecast(h+1)[-1]]) #predict h step
        predictions.append(yhat) #store prediction
        history.append(test[t]) #store observation
        # calculate out of sample error
    rmse = sqrt(mean_squared_error(test[h:], predictions))
    return rmse

def evaluate_models(dataset, p1, q1, p2, q2):
    order1 = (p1,0,q1)
    order2 = (p2,0,q2)
    rmse_list1 = []
    rmse_list2 = []
    print("Model,   1-step RMSE, 2-step RMSE, 3-step RMSE, 4-step RMSE")
    for h_val in range(4):
        rmse_list1.append(np.round(evaluate_arima_model(dataset, order1, h_val),5))
        rmse_list2.append(np.round(evaluate_arima_model(dataset, order2, h_val),5))

    print(f'Model 1: {rmse_list1[0]},     {rmse_list1[1]},     {rmse_list1[2]},     {rmse_list1[3]}')
    print(f'Model 2: {rmse_list2[0]},     {rmse_list2[1]},      {rmse_list2[2]},     {rmse_list2[3]}')
    
    
    
    
#model evaluation based on RMSE and one-step cross validation
def rmse_sarima(X, trend_order, seasonal_order, split_size): #added split_size
    train_size = int(len(X) * split_size)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    predictions = list()
    for t in range(len(test)):
        model = sm.tsa.statespace.SARIMAX(history, order=trend_order,seasonal_order=seasonal_order)
        res = model.fit()
        yhat = res.predict(start=len(history), end=len(history)) #predict one step
        predictions.append(yhat) #store prediction
        history.append(test[t]) #store observation
        # calculate out of sample error
    rmse = sqrt(mean_squared_error(test, predictions))
    return rmse

# mean_absolute_error
def mae_sarima(X, trend_order, seasonal_order, split_size): #added split_size
    train_size = int(len(X) * split_size)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    predictions = list()
    for t in range(len(test)):
        model = sm.tsa.statespace.SARIMAX(history, order=trend_order,seasonal_order=seasonal_order)
        res = model.fit()
        yhat = res.predict(start=len(history), end=len(history)) #predict one step
        predictions.append(yhat) #store prediction
        history.append(test[t]) #store observation
        # calculate out of sample error
    mae = mean_absolute_error(test, predictions)
    return mae

def mae_ETS(X, trend, seasonal, m, damped, split_size): #added split_size
    train_size = int(len(X) * split_size)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    predictions = list()
    for t in range(len(test)):
        model = ExponentialSmoothing(history, trend=trend,seasonal=seasonal, seasonal_periods=m, damped=damped)
        res = model.fit()
        yhat = res.predict(start=len(history), end=len(history)) #predict one step
        predictions.append(yhat) #store prediction
        history.append(test[t]) #store observation
        # calculate out of sample error
    mae = mean_absolute_error(test, predictions)
    return mae

def rmse_ETS(X, trend, seasonal, m, damped, split_size): #added split_size
    train_size = int(len(X) * split_size)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    predictions = list()
    for t in range(len(test)):
        model = ExponentialSmoothing(history, trend=trend,seasonal=seasonal, seasonal_periods=m, damped=damped)
        res = model.fit()
        yhat = res.predict(start=len(history), end=len(history)) #predict one step
        predictions.append(yhat) #store prediction
        history.append(test[t]) #store observation
        # calculate out of sample error
    rmse = sqrt(mean_squared_error(test, predictions))
    return rmse




def evaluate_rmse_sarima(data, p_vals, d_vals,q_vals,P_vals, D_vals, Q_vals, m_vals, split_size):
    best_score, best_order, best_seasonal_order = float("inf"), None, None
    my_dict = {'p_vals': p_vals, 'd_vals': d_vals, 'q_vals': q_vals,
               'P_vals': P_vals, 'D_vals': D_vals, 'Q_vals': Q_vals, 'm_vals':m_vals}
    
    all_comb = it.product(*(my_dict[name] for name in my_dict))
    for comb in list(all_comb):  
        order = comb[0:3]
        seasonal_order = comb[3:]
        try:
            rmse=rmse_sarima(data,trend_order=order, seasonal_order=seasonal_order, split_size=split_size)
            if rmse < best_score:
                best_score, best_order, best_seasonal_order=rmse, order,seasonal_order
        except:
            continue
    print((best_score, best_order, best_seasonal_order))
    
def mean_absolute_percentage_error(actual, pred):
    actual = np.array(actual)
    pred = np.array(pred)
    return np.sum(np.abs(actual - pred)/actual)/len(actual)


def mape_sarima(X, trend_order, seasonal_order, split_size): #added split_size
    train_size = int(len(X) * split_size)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    predictions = list()
    for t in range(len(test)):
        model = sm.tsa.statespace.SARIMAX(history, order=trend_order,seasonal_order=seasonal_order)
        res = model.fit()
        yhat = res.predict(start=len(history), end=len(history)) #predict one step
        predictions.append(yhat) #store prediction
        history.append(test[t]) #store observation
        # calculate out of sample error
    preds = [x for j in predictions for x in j]
    mape = mean_absolute_percentage_error(test, preds)
    return mape

def plot_hist_pred(history, test, pred):
    fig, ax = plt.subplots(1,2, figsize=(15,5))
    time = [history['Adj Close'], history['Adj Close']['2001-10-01':]]
    titles = ['1994-2008', '2002-2008']
    for i in range(2):
        ax[i].plot(time[i], label='Historic prices')
        ax[i].plot(test['Adj Close'],label='Actual prices')
        ax[i].plot(test[pred],label='Predicted prices')
        ax[i].legend(loc='upper left', fontsize=8)
        ax[i].set_title(titles[i])
        
def plot_hist_pred2(history, test, pred):
    fig, ax = plt.subplots(1,2, figsize=(15,5))
    time = [history['Beer_Consumption'], history['Beer_Consumption']['2015-10-01':]]
    titles = ['All Data', 'Shortened']
    for i in range(2):
        ax[i].plot(time[i], label='Beer Consumption')
        ax[i].plot(test['Beer_Consumption'],label='Actual Consumption')
        ax[i].plot(test[pred],label='Predicted Consumption')
        ax[i].legend(loc='upper left', fontsize=8)
        ax[i].set_title(titles[i])
        
def evaluate_es_model_mape(X,Trend, Seasonal,m,damped=True):
    train_size = int(len(X) * 0.8)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    predictions = list()
    for t in range(len(test)):
        model = ExponentialSmoothing(history, trend=Trend,seasonal=Seasonal,seasonal_periods=m, damped=damped)
        res = model.fit()
        yhat = res.forecast()[0] #predict one step
        predictions.append(yhat) #store prediction
        history.append(test[t]) #store observation
        # calculate out of sample error
    mape=(np.abs(np.array(test)-np.array(predictions))/np.array(test)).mean()
    return mape


def evaluate_es_model_mae(X,Trend, Seasonal,m,damped=True):
    train_size = int(len(X) * 0.8)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    predictions = list()
    for t in range(len(test)):
        model = ExponentialSmoothing(history, trend=Trend,seasonal=Seasonal,seasonal_periods=m, damped=damped)
        res = model.fit()
        yhat = res.forecast()[0] #predict one step
        predictions.append(yhat) #store prediction
        history.append(test[t]) #store observation
        # calculate out of sample error
    mae= mean_absolute_error(test, predictions)
    return mae