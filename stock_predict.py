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
from sklearn.metrics import r2_score

rnn_unit=10       #隐层数量
input_size=4
output_size=1
lr=0.0006         #学习率
f1 = open('/mnt/cephfs/lab/gaosiyi/stock/dataset/train.csv')
df1=pd.read_csv(f1)     #读入股票数据
data1 = df1.loc[:,['open','high','low','close','label']].values
#f2 = open('/mnt/cephfs/lab/gaosiyi/stock_data/600297.csv')
#df2=pd.read_csv(f2)
#data2 = df2.loc[:,['open','high','low','close','label']].values

'''
plt.plot(data[:,1],'r') #最高价
plt.plot(data[:,2],'b') #最低价
plt.plot(data[:,0],'g') #开盘价
plt.plot(data[:,3],'y') #闭盘价
plt.plot(data[:,4],'c--') #成交量
plt.plot(data[:,5],'r--')
plt.plot(data[:,6],'b--')
plt.plot(data[:,7],'y--')
plt.show()
plt.savefig("source_all.png")
'''

config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
config.gpu_options.allow_growth=True  

#获取训练集
def get_train_data(batch_size=60,time_step=20,train_begin=0,train_end=400):
    batch_index=[]
    data_train=data1[train_begin:train_end]
    normalized_train_data=(data_train-np.mean(data_train,axis=0))/np.std(data_train,axis=0)  #标准化
    train_x,train_y=[],[]   #训练集
    for i in range(len(normalized_train_data)-time_step):
       if i % batch_size==0:
           batch_index.append(i)
       x=normalized_train_data[i:i+time_step,:4]
       y=normalized_train_data[i:i+time_step,4,np.newaxis]
       train_x.append(x.tolist())
       train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data)-time_step))
    return batch_index,train_x,train_y


#获取测试集
def get_test_data(time_step=20,test_begin=400):
    data_test=data1[test_begin:]
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
    time_step=tf.shape(X)[1]
    w_in=weights['in']
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


def train_lstm(batch_size=60,time_step=20,train_begin=200,train_end=5800):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    batch_index,train_x,train_y=get_train_data(batch_size,time_step,train_begin,train_end)
    with tf.variable_scope("sec_lstm",reuse=tf.AUTO_REUSE):
        pred,_=lstm(X,'train')
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    saver=tf.train.Saver(tf.global_variables(),max_to_keep=15)
    #module_file = tf.train.latest_checkpoint() 

    with tf.Session(config = config) as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1000):     #这个迭代次数，可以更改，越大预测效果会更好，但需要更长时间
            for step in range(len(batch_index)-1):
                _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]]})
            print("Number of iterations:",i," loss:",loss_)
            if i % 200 == 0:
                print("model_save: ",saver.save(sess,'model_save/stock.model',global_step = i))
        print("The train has finished")

def prediction(time_step=20):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    mean,std,test_x,test_y=get_test_data(time_step)
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
        #rmse = np.sqrt(sum((test_predict-test_y[:len(test_predict)])**2)/test_y.size)
        score = r2_score(test_y[:len(test_predict)],test_predict)
        print("The accuracy of this predict:",score)
        #以折线图表示结果
        plt.figure()
        plt.plot(list(range(len(test_predict))), test_predict, color='b',label = 'Predict')
        plt.plot(list(range(len(test_y))), test_y,  color='r',label='Original')
        plt.title('score:%f' % score)
        #plt.show()
        plt.savefig("stock.jpg")

if __name__ == "__main__":
    #train_time = time.time()
    #train_lstm()
    #print time.time()-train_time
    #predict_time = time.time()
    prediction()
    #print time.time()-predict_time

