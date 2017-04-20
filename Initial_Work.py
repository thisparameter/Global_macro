# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import sklearn as sk

def gather_data(filename, sheet):
    df = pd.read_excel(filename,sheetname=sheet)
    #drop columns where all the values are na
    df = df.dropna(axis = 1, how = 'all')
    #naming the columns properly
    return df

def process_headers(df):
    df = df.copy()
    curr_colnames = list(df.columns)
    colnames =list(df.loc[0,:])
    for i,col in enumerate(list(curr_colnames)):
        if col.find("Unnamed") != -1:
            curr_colnames[i] = curr_colnames[i-1]
            col = curr_colnames[i-1]
        colnames[i]= str(col) + '__'+str(colnames[i]) 
    df.columns = colnames
    df = df.loc[1:,:]
    return df
        
def process_timeline(df):
    #aligns the dataframe with single timeline
    #break the dataframe into separate dataframes with regards to timelines
    df_dict = {}
    count_dates = [0 if x.lower().find('date')==-1 else 1 for x in list(df.columns)]
    
    ct_df = 0
    start = 0
    start_index = 0
    for i,dt in enumerate(count_dates):
        if dt == 1 and start ==0:
            df_dict[ct_df] = pd.DataFrame()
            start_index = i
            start = 1
            
        elif dt ==1:
            df_dict[ct_df] = df.iloc[:,start_index:i]
            ct_df = ct_df + 1
            start_index = i
            
        elif i == len(count_dates) - 1 and start == 1:
            df_dict[ct_df] = df.iloc[:,start_index:]
            start = 0
    
    for key in sorted(list(df_dict.keys())):
        temp_df = df_dict[key].copy()
        temp_df = temp_df.loc[temp_df.iloc[:,0].dropna().index,:]
        temp_df.index = temp_df.iloc[:,0]
        temp_df = temp_df.iloc[:,1:]
        df_dict[key] = temp_df
        
    #merging the dataframes together
    total_df = np.nan
    for i,key in enumerate(sorted(list(df_dict.keys()))):
        if i == 0:
            total_df =df_dict[key]
        else:
            #print key, df_dict[key].head()
            total_df = total_df.merge(df_dict[key],left_index = True, right_index = True, how = 'outer')
    return total_df
    pass


def fly_calc(df, securities):
    #calculates the flies on 1-2-1 basis and gets the time-series of the fly
    pass
    return -1*np.array(df[securities[0]]) + 2*np.array(df[securities[1]]) + -1*np.array(df[securities[2]])

def fly_pca_calc(df, securities):
    temp_df = df[securities].dropna()
    X1 = np.array(temp_df[securities[0]])
    X2 = np.array(temp_df[securities[1]])
    X3 = np.array(temp_df[securities[2]])
    X = np.array(temp_df[securities])
    pca_model = PCA()
    pca_model.fit(X)
    weights = pca_model.components_[:,2]
    #normalize the weights
    weights = weights/weights[1]*2.
    #calc the fly
    return weights[0]*df[securities[0]]+weights[1]*df[securities[1]]+weights[2]*df[securities[2]]
    pass

#mean reversion equation - dxt = a(b-xt-1) + error
import sklearn as sk
from sklearn import linear_model
from sklearn.decomposition import PCA

def mean_reversion(flies):
    model_gaussian = linear_model.LinearRegression()
    dt_flies = flies - flies.shift(1)
    X = np.array(flies.shift(1).dropna()).reshape((-1,1))
    Y = np.array(dt_flies.dropna()).reshape((-1,1))
    model_gaussian.fit(X = X, y=Y)
    alpha = model_gaussian.coef_[0]*-1.
    theta = model_gaussian.intercept_/alpha
    var = model_gaussian.residues_
    return list(alpha)[0], list(theta)[0], list(var)[0]
    pass