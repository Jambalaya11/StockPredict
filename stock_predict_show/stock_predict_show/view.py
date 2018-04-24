from __future__ import unicode_literals 
from django.http import HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from random import choice
import json
import urllib2
import os
import csv
#from predict import get_predict_result
 
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = '/mnt/cephfs/lab/gaosiyi/stock_data_predict'
info_path = os.path.join(CURRENT_DIR,'dataset/codes_info.csv')

code_info = {}
with open(info_path,"r") as infofile:
    items = csv.reader(infofile)
    for i,obj in enumerate(items):
        if i >= 1:
            code_info[obj[1]] = [obj[2],obj[3],[],[],[]]
files = os.listdir(data_path)
for ofile in files:
    code = ofile.split('.')[0]
    if code in code_info.keys():
        ofile = open(os.path.join(data_path,ofile),'r')
        reader = csv.reader(ofile)
        for j,lines in enumerate(reader):
            if j >= 1:
                code_info[code][2].append(lines[0])
                code_info[code][3].append(lines[5])
                code_info[code][4].append(lines[6])


def index(request):
    return render(request, 'index.html', {})

def cluster(request):
    return render(request, 'cluster.html',{})

def predict(request):
    context = {}
    context['c_name'] = []
    for code,info in code_info.items():
        name = info[0]
        c_name = info[1]
        context.setdefault(c_name,[]).append(name) 
    return render(request, 'predict.html', context)

def method(request):
    return render(request, 'method.html', {})

@csrf_exempt
def get_cluster_result(request):
    context= {"msg":"success", "code":0}
    task_name = request.POST.get("task_name", "")
    #print task_name
    context['date'] = []
    context['data'] = []
    context['code'] = []
    context['name'] = []
    for code,info in code_info.items():
        name = info[0]
        c_name = info[1]
        if len(info[2]) != 0:
            date = info[2]
            data = info[3]
            #print date,data
            if unicode(c_name.decode('utf8')) == task_name:
                context['name'].append(name)
                context['date'].append(date)
                context['code'].append(code)
                context['data'].append(data)
    return HttpResponse(json.dumps(context))

@csrf_exempt
def get_predict_result(request):
    context= {"msg":"success", "code":0}
    res_name = request.POST.get("res_name", "")
    for code,info in code_info.items():
        ori_name = info[0]
        if unicode(ori_name.decode('utf8')) == res_name:
            context['date'] ,context['data'] ,context['data_pre']= info[2],info[3],info[4]
    return HttpResponse(json.dumps(context))
            
        
    
