# -*- coding: utf-8 -*-
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def FileCheck(filename):
    cur_dir = os.getcwd()
    cur_dir = cur_dir.replace(chr(92), '/') + '/'
    if Path(cur_dir + filename).exists():
        return True
    else:
        return False


def DataLoading():
    # data loading
    if FileCheck('data_altman.h5'):
        data = pd.read_hdf('data_altman.h5', 'table')
    else:
        data = pd.read_excel('data_altman.xlsx', sheetname='merge')
        # data preprocessing
        data = data.iloc[1:]  # 첫째행 버림

        data.to_hdf('data_altman.h5', 'table')
    data.set_index(['year', 'ticker', 'company'])
    y = data['y'].astype(np.float64)
    x = data[data.columns[4:34]].astype(np.float64)
    others = data[data.columns[34:-1]]
    end = data[data.columns[-1]].astype(np.float64)
    ret = pd.concat([y, x, others, end], axis=1)
    return ret


def AltmanZ():
    '''Altman found that the ratio profile for the bankrupt group fell at −0.25 avg,
    and for the non-bankrupt group at +4.48 avg.'''
    z1 = (data['x07'] - data['x09']) / data['x06']  # 운전자본/총자산
    z2 = data['x13'] / data['x06']  # 이익잉여금/총자산
    z3 = data['x18'] / data['x06']  # 영업이익/총자산
    z4 = data['x24'] / data['x08']  # 시가총액/총부채
    z5 = data['x15'] / data['x06']  # 매출액/총자산
    z = 1.2 * z1 + 1.4 * z2 + 3.3 * z3 + 0.6 * z4 + 1.0 * z5
    return z


def OhlsonO():
    '''For the O-Score, any results larger than 0.5 suggests that the firm will default within two years.'''
    t1 = data['x06'] / data['x30']  # 총자산/GNP price indes (없어서 소비자물가지수 사용)
    t2 = data['x08'] / data['x06']  # 총부채/총자산
    t3 = (data['x07'] - data['x09']) / data['x06']  # 운전자본/총자산
    t4 = data['x09'] / data['x07']  # 유동부채/유동자산
    t5 = data['x06'] < data['x08']  # 자본잠식 1, otherwise 0
    t6 = data['x19'] / data['x06']  # 당기순이익/총자산
    t7 = data['x21'] / data['x08']  # 영업활동으로인한현금흐름/총부채
    t8 = data['x32'] < 0
    t9 = (data['x19'] - data['x32']) / (np.abs(data['x19']) + np.abs(data['x32']))
    T = -1.32 - 0.407 * np.log(
        t1) + 6.03 * t2 - 1.43 * t3 + 0.0757 * t4 - 1.72 * t5 - 2.37 * t6 - 1.83 * t7 + 0.285 + t8 - 0.521 * t9
    o = np.exp(T) / (1 + np.exp(T))
    return o


def Cutoff_Fitting(y, x, minimum, maximum, step, model='Altman'):
    result = []
    for i in np.arange(minimum, maximum, step):
        if model == 'Altman':
            predict = x < i
        else:
            predict = x > i
        TP = sum(y * predict)  # 실제 1을 1이라고 예측한것
        TN = sum((y - 1) * (predict - 1))  # 실제 0을 0이라고 예측한것
        FP = -1 * sum((y - 1) * predict)  # 실제 0을 1이라고 예측한것
        FN = -1 * sum(y * (predict - 1))  # 실제 1을 0이라고 예측한것
        Precision = TP / (TP + FP + 1e-20)  # 1이라고 예측한것 중 실제 1인것의 비중
        Recall = TP / (TP + FN + 1e-20)  # 실제 1인것들 중에서 예측결과가 1인 것의 비중
        Accuracy = (TP + TN) / (TP + TN + FP + FN)  # 정확히 예측(즉, 1을 1이라고, 0을 0이라고 예측)한 것의 비중
        ClassificationError = (FP + FN) / (TP + TN + FP + FN)  # 틀리게 예측(즉, 1을 0이라고, 0을 1이라고 예측)한 것의 비중
        F1score = 2 / (1 / Precision + 1 / Recall + 1e-20)  # harmonic mean
        tmp = [i, TP, FP, FN, TN, Precision, Recall, Accuracy, ClassificationError, F1score]
        result.append(tmp)
    res_col = ['Cutoff', 'True Positive', 'False Positive', 'False Negative', 'True Negative', 'Precision', 'Recall',
               'Accuracy', 'Classification Error', 'F1 Score']
    result = pd.DataFrame(result, columns=res_col)
    return result


data = DataLoading()
data = data.dropna()

y = data['y']
x_altmanz = AltmanZ()
x_ohlsono = OhlsonO()

result_altman = Cutoff_Fitting(y, x_altmanz, -10, 10, 0.01)
result_ohlson = Cutoff_Fitting(y, x_ohlsono, 0.01, 1.01, 0.01, 'Ohlson')

# plot
plt.figure(0)
plt.title('Altman Z-score Performance')
plt.plot(result_altman['Cutoff'], result_altman['Accuracy'], label="Accuracy")
plt.plot(result_altman['Cutoff'], result_altman['Classification Error'], label="Classification Error")
plt.plot(result_altman['Cutoff'], result_altman['F1 Score'], label="F1 Score")
plt.legend()

plt.figure(1)
plt.title("Ohlson's O-score Performance")
plt.plot(result_ohlson['Cutoff'], result_ohlson['Accuracy'], label="Accuracy")
plt.plot(result_ohlson['Cutoff'], result_ohlson['Classification Error'], label="Classification Error")
plt.plot(result_ohlson['Cutoff'], result_ohlson['F1 Score'], label="F1 Score")
plt.legend()
