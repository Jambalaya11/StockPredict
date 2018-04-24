#coding=utf-8
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
from statsmodels.tsa.arima_model import ARMA

def draw(data,scope):
    plt.figure()
    plt.plot(data) #最高价
    plt.show()
    plt.savefig(scope+'.png')

if __name__ == '__main__':
    f1 = open('train.csv')
    df1=pd.read_csv(f1)     #读入股票数据
    data = df1.loc[:,['open','high','low','close','label']].values
    data = data[:,1]
    #draw(data,'origin')
    diff = pd.Series(data).diff(1) #reduce the influence of time to data
    #draw(diff,'diff')
    fig = plt.figure(figsize=(12,8))
    ax1=fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(data,lags=40,ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(data,lags=40,ax=ax2)
    fig.savefig('pacf.png')





