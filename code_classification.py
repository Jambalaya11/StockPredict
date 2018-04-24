#coding=utf-8
import csv
import collections
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(CURRENT_DIR,'dataset/rowData.csv')
code_path = os.path.join(CURRENT_DIR,'dataset/codes_info.csv')

def read_csv(file):
	codelist = []
	with open(file,"r") as df:
		reader = csv.reader(df)
		for i,line in enumerate(reader):
			if i>=1:
				codelist.append(line)
	return codelist

if __name__ == '__main__':
	codelist = read_csv(data_path)
	infolist = read_csv(code_path)
	infoclass = {}
	classlist = []
	for item in codelist:
		code = item[-1]
		for j in infolist:
			if code == j[1]:
				classlist.append(j[3])
				#infoclass[j[3]] += 1
	d = collections.Counter(classlist)
	print d.keys()
