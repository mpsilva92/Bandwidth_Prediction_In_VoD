from __future__ import division
# from tensorflow.keras import backend as K
from tensorflow.keras import backend as K
# from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statistics
import argparse
import logging
import pickle
import os

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s:%(levelname)s:%(message)s")
log = logging.getLogger(__name__)
# log.setLevel(logging.ERROR)

# error metrics
# from sklearn.neighbors import _dist_metrics
from sklearn.metrics import mean_squared_error
# from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from math import sqrt

from util import Metrics, PreARTrans, PostARTrans, LoadModel, DataPrepare


custom_objects = {
    'corr': Metrics.corr,
    'rse': Metrics.rse,
    'mae': Metrics.mean_absolute_error,
    'mse': Metrics.mean_squared_error,
    'PreARTrans': PreARTrans,
    'PostARTrans': PostARTrans
}


def data_plot(init, predSet, testSet):
    if predSet is not None and testSet is not None:
        save_path = './plots/prediction_of_'+init.model+'.pdf'
        fig = plt.figure()
        plt.plot(testSet.flatten()[0:30], color='blue', label='original')
        plt.plot(predSet.flatten()[:30], 'k--', color='red', label='prediction')
        plt.xlabel('Time [s]')
        plt.ylabel('Throughput in Mbit/s')
        plt.legend(loc='best')
        plt.grid(True)
        # log.debug("Saving plot to: s%", save_path)
        plt.show();
        fig.savefig(save_path)

        
def evaluate_model(init, data, predict):
    # start and end of each session
    START = 0
    END = 5
    
    # create empty list
    mse_store  = []
    mae_store  = []
    rmse_store = []
    mape_store = []
    rse_store  = []
    
    if predict is not None:
        
        for index in range(len(data)):
            orig_data = data.flatten()[START:END]
            pred_data = predict.flatten()[START:END]
            
            # evaluating with error metrics
            mse = mean_squared_error(orig_data, pred_data)
            mae = mean_absolute_error(orig_data, pred_data)
            rmse = sqrt(mean_squared_error(orig_data, pred_data))
            mape = mean_absolute_percentage_error(orig_data, pred_data)
            rse_m = Metrics.rse(K.constant(orig_data), K.constant(pred_data))
            
            mse_store.append(mse)    # mean squared error
            mae_store.append(mae)    # mean absolute error
            rmse_store.append(rmse)  # root mean squared error
            mape_store.append(mape)  # Mean absolute percentage error (MAPE) regression loss
            rse_store.append(rse_m.numpy())  # computes the relative squared error
            
            START+=5
            END+=5
            
        # inicialize an empty dict with metrics
        metrics_store =  {'MSE':mse_store, 'MAE':mae_store, 'RMSE':rmse_store, 'RSE':rse_store, 'MAPE': mape_store}
        # transform in dataframe
        metrics_df = pd.DataFrame(metrics_store)
        # save the dataframe in mode csv
        metric_name = init.data.split("/")
        metric_name = metric_name[-1][:-4]
        metrics_df.to_csv('./results/test/METRIC_'+init.model+metric_name+'.csv', encoding='utf-8', index=False)
    else:
        exit(1)
    
        
def exponentially_weighted_moving_average(data):
    throughput        = np.array(data.dataLoad[:, -1])  # column last (throughput)
    EWMA = []                                           # initialize an empty list to store cumulative EWMA of each session
    store_throughput  = []
    arr_array         = []
    window            = 10 # sliding window (past)
    horizon           = 5  # future
    
    start, end = 0, throughput.shape[0]
    start = start + window
    end = end - horizon
    
    for count in range(start, end):
        index = range(count+1, count+1+horizon)
        store_throughput.append(throughput[index])
        # store_throughput.append(throughput[count:count+window])    
        
    # array and dataframe
    store_throughput = np.array(store_throughput)
    df = pd.Series(store_throughput.flatten())
    
    # EWMA: calculate the cumulative EWMA using pandas.Series.rolling method
    count = 0
    while (count < len(df)):
        ewma = df[count:count+horizon].ewm(span=window, adjust=False).mean()
        EWMA.append(ewma)
        # arr_array.append(EWMA)
        count+=5
    return np.array(EWMA)


def harmonic_mean(data):
    throughput        = data.dataLoad[:, -1]            # column last (throughput)
    # harmonicMean      = []                            # initialize an empty list to store Harmonic Mean
    store_throughput  = []
    arr_array         = []
    window            = 10                              # sliding window
    horizon           = 5
    
    # count = 0
    start, end = 0, throughput.shape[0]
    start = start + window
    end = end - horizon
    
    for count in range(start, end):
        index = range(count+1, count+1+horizon)
        store_throughput.append(throughput[index])
        # store_throughput.append(throughput[count:count+window])
        
    k = 0
    while(k < len(store_throughput)):
        harm_mean = store_throughput[k]
        j = 0
        harmonicMean = []
        while(j < len(harm_mean)):
            hm_value = round(statistics.harmonic_mean(harm_mean[:j+1]), 8)
            harmonicMean.append(hm_value)
            j+=1
        arr_array.append(harmonicMean)
        k+=1
    return np.array(arr_array)


def simple_moving_average(data):

    throughput        = data.dataLoad[:, -1]       # column last (throughput)
    # moving_averages   = []                       # initialize an empty list to store moving averages
    store_throughput  = []
    arr_array         = []
    window            = 10                          # sliding window (past)
    horizon           = 5                           # future
    
    # count = 0
    start, end = 0, throughput.shape[0]
    start = start + window
    end = end - horizon
    
    # throughput = throughput.reshape(-1)
    for count in range(start, end):
        index = range(count+1, count+1+horizon)
        store_throughput.append(throughput[index])
        # store_throughput.append(throughput[count:count+window])
        
    k = 0
    while(k < len(store_throughput)):
        cum_sum = store_throughput[k]
        i = 1
        moving_averages = []
        while(i <= len(cum_sum)):
            window_average = round(cum_sum[i-1] / i, 8)
            moving_averages.append(window_average)
            i+=1
        arr_array.append(moving_averages)
        k+=1
    return np.array(arr_array)


def z_score_invert(init, Data, predict):
    if predict is not None:
        get_throughput = Data.dataLoad[:, -1]  # column last (throughput)
        end = Data.data.shape[0]
        window  = 10
        horizon = 5
        start   = 0 
        start = start + window
        end = end - horizon
        testData = []
        testPred = []
        
        if init.model=='GRU' or init.model=='LSTM' or init.model=='BiLSTM' or init.model=='CNNGRU' or init.model=='CNNLSTM' or init.model=='CNNBiLSTM' or init.model=='CNNBiGRU' or init.model == 'SMA' or init.model == 'HM' or init.model == 'EWMA':
            # lookback
            for count in range(start, end):
                index = range(count+1, count+1+horizon)
                testData.append(get_throughput[index])
            testPredict = (np.array(testData) + np.mean(predict)) * np.std(predict)
            testData = np.array(testData)
            
        if init.model=='RF' or init.model=='KNN' or init.model=='SVR':
            # lookback
            for count in range(start, end):
                index = range(count+1, count+1+horizon)
                testData.append(get_throughput[index])
                testPred.append(predict[index])
            testPredict = (np.array(testData) + np.mean(np.array(testPred))) * np.std(np.array(testPred))
            testData = np.array(testData)
        
        if init.approach == 'DL' or init.approach == 'ML':
            # evaluate the model with metrics 
            evaluate_model(init, testData, testPredict)

            # plot the prediction of the model on test set
            data_plot(init, testPredict, testData)
        return np.array(testData), testPredict


def model_execute(init):
    # w: window
    # h: horizon
    # DL: path to RNN model
    # ML: path to ML model
    if init.data is not None and init.model is not None:        
        w       =     10
        h       =     5
        FLAG    =     True
        
        if init.approach == 'DL': 
            if init.model=='GRU' or init.model=='LSTM' or init.model=='BiLSTM' or init.model=='CNNGRU' or init.model=='CNNLSTM' or 'CNNBiLSTM' or init.model=='CNNBiGRU':
                filepath = init.path + 'DL/' + init.model  + '/' + init.model # path to model file
                log.info("Load model from %s", os.path.dirname(os.path.abspath(filepath))) 
                model = LoadModel(filepath, custom_objects) # load RNN model
                print(model.summary())
                Data = DataPrepare(init.data, w, h, FLAG)

                # get prediction
                testPredict = model.predict(Data.X_test[0])
                orig_data, pred_data = z_score_invert(init, Data, testPredict) # invert data

                print('\nMSE:', mean_squared_error(orig_data, pred_data))
                print('\nMAE:', mean_absolute_error(orig_data, pred_data))
                print('\nRMSE:', np.sqrt(mean_squared_error(orig_data, pred_data)))
                print('\nRSE:', Metrics.rse(K.constant(orig_data), K.constant(pred_data)))
            
            else:
                raise TypeError("Model is not defined")
        
        elif init.approach == 'ML': 
            if init.model=='RF' or init.model=='SVR' or init.model=='KNN':
                FLAG=False
                filepath = init.path + 'ML/' + init.model + '.pkl' # path to model file
                log.info("Load model from %s", os.path.dirname(os.path.abspath(filepath)))
                try:
                    # load model from filepath
                    with open(filepath, 'rb') as file:
                        model = pickle.load(file)
                    print(model)
                except IOError as err:
                    log.error('Fail to load model')
                Data = DataPrepare(init.data, w, h, FLAG)
                # print(Data.data.shape)
            else:
                raise TypeError("Model is not defined")
                
            testData = Data.data[:, :-1]
            testPredict = model.predict(testData)                            # get prediction
            orig_data, pred_data = z_score_invert(init, Data, testPredict)   # invert data
            
            # print('real data', orig_data.shape)
            # print('predicted data', pred_data.shape)
            
            print('\nMSE:', mean_squared_error(orig_data, pred_data))
            print('\nMAE:', mean_absolute_error(orig_data, pred_data))
            print('\nRMSE:', np.sqrt(mean_squared_error(orig_data, pred_data)))
            print('\nRSE:', Metrics.rse(K.constant(orig_data), K.constant(pred_data)))
        
        else: 
            if init.approach == 'MA':
                # model for comparison
                # DEEP_MODEL = 'CNNBiGRU'
                
                # prepare input data
                Data = DataPrepare(init.data, w, h, FLAG)
                
                # load neurak network model
                # filepath = init.path + 'DL/' + DEEP_MODEL + '/' + DEEP_MODEL                   # path to model file
                # log.info("Load model from %s", os.path.dirname(os.path.abspath(filepath))) 
                # model = LoadModel(filepath, custom_objects)
                # print(model.summary())

                # get prediction
                # testPredict = model.predict(Data.X_test[0])
                # print('\npredicted data:', len(testPredict))
                
                # orig_data, pred_data = z_score_invert(init, Data, testPredict) # invert data
                
                # Simple Moving Average [SMA]
                if init.model == 'SMA':
                    data_of_sma = simple_moving_average(Data)
                    # evaluate_model(init, data_of_sma, pred_data)
                    evaluate_model(init, Data.dataLoad[:, -1], data_of_sma)
                    
                # Harmonic Mean [HM] 
                elif init.model == 'HM':
                    data_of_hm = harmonic_mean(Data)
                    evaluate_model(init, Data.dataLoad[:, -1], data_of_hm)
                
                # Exponentially Weighted Moving Average [EWMA]
                else:
                    if init.model == 'EWMA':
                        data_of_ewma = exponentially_weighted_moving_average(Data)
                        evaluate_model(init, Data.dataLoad[:, -1], data_of_ewma)
                    

if __name__== '__main__':
    # log.info('Worked')
    try:
        p=argparse.ArgumentParser(description='Throughput Prediction Model in Video Streaming')
        p.add_argument('--data', type=str, required=True, help='File location')
        p.add_argument('--model', type=str, choices=['GRU','LSTM','BiLSTM','CNNGRU','CNNLSTM','CNNBiLSTM', 'CNNBiGRU', 'RF','KNN','SVR','SMA', 'HM', 'EWMA', None], default=None, help='The model to get prediction. Default None')
        p.add_argument('--path', type=str, default='./results/training/', help='Load the model from origem path')
        p.add_argument('--approach', required=True, type=str, default='DL', help='Model from ML or DL => Machine Learning or Deep Learning)')
        args = p.parse_args()
        model_execute(args)
    except SystemExit as err:
        print('Invalid argument')
