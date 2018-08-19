# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 19:58:16 2017

@author: ram
"""

import pandas as pd



columns_to_keep=['age','sex','chst_pain_type','resting_blood_prs',
           'serum_chl_mg_dl','fasting_blood_sugar','resting_electrocardiographic_results',
           'maximum_heart_rate_achieved','exercise_induced_angina',
           'oldpeak','slope_peak_exer_ST','number_of_major_vessels','thal','heart_disease']

url='http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat'

#df = pd.read_table(url, sep="\s+", names=columns_to_keep,index_col=False)
df=pd.read_csv("D:\\Ram\\Kaggle\\Heart Disease\\train.csv",header=0)
df.head()

print('Data has {} no. of rows and {} no. of attributes'.format(df.shape[0],df.shape[1]))

print('Data has below columns with missing rows\n{} '.format(df.isnull().sum()))

df.dtypes

#Lets plot each attribute one by one and try to find out some relationship.

#if we go in order then we can start our expolaratory data analysis with age and sex.
    #1. First check number of information in sex column
    df.sex.value_counts()  #sex: sex (1 = male; 0 = female) 
    #2.  #age: age in years
    grouped = data['2013-08-17'].groupby(axis=1, level='SPECIES').T
grouped.boxplot()
data['2013-08-17'].boxplot(by='SPECIES')
data.boxplot(column='2013-08-17',by='SPECIES')







