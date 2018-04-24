#coding=utf-8
import matplotlib
matplotlib.use('Agg')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import csv

rnn_unit=10       #隐层数量
input_size=4
output_size=1
lr=0.0006         #学习率

config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
config.gpu_options.allow_growth=True


def get_test_data(data,time_step=20,test_begin=500):
    data_test=data
    mean=np.mean(data_test,axis=0)
    std=np.std(data_test,axis=0)
    normalized_test_data=(data_test-mean)/std  #标准化
    size=(len(normalized_test_data)+time_step-1)//time_step  #有size个sample
    test_x,test_y=[],[]
    for i in range(size-1):
       x=normalized_test_data[i*time_step:(i+1)*time_step,:4]
       y=normalized_test_data[i*time_step:(i+1)*time_step,4]
       test_x.append(x.tolist())
       test_y.extend(y)
    test_x.append((normalized_test_data[(i+1)*time_step:,:4]).tolist())
    test_y.extend((normalized_test_data[(i+1)*time_step:,4]).tolist())
    return mean,std,test_x,test_y

weights={
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random_normal([rnn_unit,1]))
        }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))
       }

def lstm(X,scope):

    batch_size=tf.shape(X)[0]
    time_step = tf.shape(X)[1]
    w_in = weights['in']
    b_in=biases['in']
    input=tf.reshape(X,[-1,input_size])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  #将tensor转成3维，作为lstm cell的输入
    cell=tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    init_state=cell.zero_state(batch_size,dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)
    output=tf.reshape(output_rnn,[-1,rnn_unit])
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states

def prediction(data,time_step=20):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    mean,std,test_x,test_y=get_test_data(data,time_step)
    with tf.variable_scope("sec_lstm",reuse=tf.AUTO_REUSE):
        pred,_=lstm(X,'pred')
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session(config=config) as sess:
        #获取最后一次保存的模型
        module_file = tf.train.latest_checkpoint('model_save/')
        saver.restore(sess, module_file)
        test_predict=[]
        for step in range(len(test_x)-1):
          prob=sess.run(pred,feed_dict={X:[test_x[step]]})
          predict=prob.reshape((-1))
          test_predict.extend(predict)
        test_y=np.array(test_y)*std[4]+mean[4]
        test_predict=np.array(test_predict)*std[4]+mean[4]
        #acc=np.average(np.abs(test_predict-test_y[:len(test_predict)])/test_y[:len(test_predict)])
        #acc = 1-acc
        rmse = np.sqrt(sum((test_predict-test_y[:len(test_predict)])**2)/test_y.size)
        print("rmse",rmse)
        return test_x,test_y,test_predict
        #print("The accuracy of this predict:",rmse)

if __name__ == '__main__':
#def get_predict_result(code):
    dir_path = '/mnt/cephfs/lab/gaosiyi/stock_data'
    des_path = '/mnt/cephfs/lab/gaosiyi/stock_data_predict'
    files= os.listdir(dir_path)
    for ofile in files:
        fname = os.path.join(dir_path,ofile)
        f = open(fname)
        df=pd.read_csv(f)
        data = df.loc[:,['open','high','low','close','label']].values
        date = df.loc[:,'date']
        x,y,y1 = prediction(data)
        resfile = os.path.join(des_path,ofile)
        print resfile
        with open(resfile,'w') as f1:
            writer = csv.writer(f1)
            writer.writerow(['date','open','high','low','close','label','result'])
            for i,row in enumerate(data):
                if i < len(y1):
                    row = list(row)
                    row.insert(0,date[i])
                    row.append(y1[i])
                    row = np.array(row)
                    writer.writerow(row)

