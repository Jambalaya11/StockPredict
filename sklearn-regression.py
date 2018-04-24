# -*-coding:utf-8-*-
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sys
import pandas as pd

###########1.数据生成部分##########

df = pd.read_csv('600048.csv')
data = df.loc[:,['open','high','low','close','label']].values
print len(data)
data_train = data[0:400]
data_test = data[400:]


def get_train_data():
    normalized_train_data=(data_train-np.mean(data_train,axis=0))/np.std(data_train,axis=0)
    x_train=normalized_train_data[:,:4]
    y_train=normalized_train_data[:,4,np.newaxis] 
    return x_train,y_train

def get_test_data():
    mean=np.mean(data_test,axis=0)
    std=np.std(data_test,axis=0)
    normalized_test_data=(data_test-mean)/std  #标准化
    x_test=normalized_test_data[:,:4]
    y_test=normalized_test_data[:,4]
    return x_test,y_test



###########2.回归部分##########
def try_different_method(model,scope):
    x_train,y_train = get_train_data()
    x_test,y_test = get_test_data()
    model.fit(x_train,y_train)
    score = model.score(x_test, y_test)
    result = model.predict(x_test)
    print score
    plt.figure()
    plt.plot(np.arange(len(result)), y_test,'go-',label='true value')
    plt.plot(np.arange(len(result)),result,'ro-',label='predict value')
    plt.title('score: %f'%score)
    plt.legend()
    #plt.show()
    plt.savefig(scope+'.jpg')

###########3.具体方法选择##########
####3.1决策树回归####
from sklearn import tree
model_DecisionTreeRegressor = tree.DecisionTreeRegressor()
####3.2线性回归####
from sklearn import linear_model
model_LinearRegression = linear_model.LinearRegression()
####3.3SVM回归####
from sklearn import svm
model_SVR = svm.SVR()
####3.4KNN回归####
from sklearn import neighbors
model_KNeighborsRegressor = neighbors.KNeighborsRegressor()
####3.5随机森林回归####
from sklearn import ensemble
model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=20)#这里使用20个决策树
####3.6Adaboost回归####
from sklearn import ensemble
model_AdaBoostRegressor = ensemble.AdaBoostRegressor(n_estimators=50)#这里使用50个决策树
####3.7GBRT回归####
from sklearn import ensemble
model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(n_estimators=100)#这里使用100个决策树
####3.8Bagging回归####
from sklearn.ensemble import BaggingRegressor
model_BaggingRegressor = BaggingRegressor()
####3.9ExtraTree极端随机树回归####
from sklearn.tree import ExtraTreeRegressor
model_ExtraTreeRegressor = ExtraTreeRegressor()


###########4.具体方法调用部分##########
try_different_method(model_GradientBoostingRegressor,'GradientBoostingRegressor')