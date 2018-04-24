#coding=utf-8
import matplotlib
import csv
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import os

source_file = '/mnt/cephfs/lab/gaosiyi/stock/dataset/600848.csv'
dest_dir = '/mnt/cephfs/lab/gaosiyi/stock_data'


def write_to_file(data):
    #file_path = os.path.join(dest_dir,str(code)+ '.csv')
    #data_trans = np.zeros((row-1,5))
    with open('600848.csv','w') as cf:
        writer = csv.writer(cf)
        writer.writerow(['date','open','high','low','close','label'])
        for i in range(len(data)-1):
            high = data[i+1][1]
            data_list = list(data[i])
            data_list.append(high)
            writer.writerow(data_list)
            #data_trans[i] = np.array(data_list)
        last_data = list(data[-1])
        last_data.append(last_data[1])
        #data_trans[i] = np.array(last_data)
        writer.writerow(last_data)


if __name__ == '__main__':
    df = pd.read_csv(source_file)   #dataFrame
    '''
    data = df.loc[:,['date','open','high','low','close','code']].values
    code_list = data[:,-1]
    raw_dict = {}
    last = 0
    for i in range(len(code_list)-1):
        if code_list[i] != code_list[i+1]:
            raw_dict[str(code_list[i])] = data[last:i+1,:5]
            last = i+1
    for code,stocks in raw_dict.items():
        code = "%06d" % int(code)
        print code
        row = len(stocks)
        write_to_file(code,stocks,row)
    '''
    data = df.loc[:,['date','open','high','low','close']].values
    write_to_file(data)

    
     

