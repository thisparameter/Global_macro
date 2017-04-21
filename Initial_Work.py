# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import sklearn as sk
#mean reversion equation - dxt = a(b-xt-1) + error
import sklearn as sk
from sklearn import linear_model
from sklearn.decomposition import PCA


def gather_data(filename, sheet):
    df = pd.read_excel(filename, sheetname=sheet)
    # drop columns where all the values are na
    df = df.dropna(axis=1, how='all')
    # naming the columns properly
    return df


def process_headers(df):
    df = df.copy()
    curr_colnames = list(df.columns)
    colnames = list(df.loc[0, :])
    for i, col in enumerate(list(curr_colnames)):
        if col.find("Unnamed") != -1:
            curr_colnames[i] = curr_colnames[i - 1]
            col = curr_colnames[i - 1]
        colnames[i] = str(col) + '__' + str(colnames[i])
    df.columns = colnames
    df = df.loc[1:, :]
    return df


def process_timeline(df):
    # aligns the dataframe with single timeline
    # break the dataframe into separate dataframes with regards to timelines
    df_dict = {}
    count_dates = [0 if x.lower().find('date') == -1 else 1 for x in list(df.columns)]

    ct_df = 0
    start = 0
    start_index = 0
    for i, dt in enumerate(count_dates):
        if dt == 1 and start == 0:
            df_dict[ct_df] = pd.DataFrame()
            start_index = i
            start = 1

        elif dt == 1:
            df_dict[ct_df] = df.iloc[:, start_index:i]
            ct_df = ct_df + 1
            start_index = i

        elif i == len(count_dates) - 1 and start == 1:
            df_dict[ct_df] = df.iloc[:, start_index:]
            start = 0

    for key in sorted(list(df_dict.keys())):
        temp_df = df_dict[key].copy()
        temp_df = temp_df.loc[temp_df.iloc[:, 0].dropna().index, :]
        temp_df.index = temp_df.iloc[:, 0]
        temp_df = temp_df.iloc[:, 1:]
        df_dict[key] = temp_df

    # merging the dataframes together
    total_df = np.nan
    for i, key in enumerate(sorted(list(df_dict.keys()))):
        if i == 0:
            total_df = df_dict[key]
        else:
            # print key, df_dict[key].head()
            total_df = total_df.merge(df_dict[key], left_index=True, right_index=True, how='outer')
    return total_df
    pass


def fly_calc(df, securities):
    #calculates the flies on 1-2-1 basis and gets the time-series of the fly
    pass
    return -1*(df[securities[0]]) + 2*(df[securities[1]]) + -1*(df[securities[2]])

def fly_pca_calc(df, securities):
    temp_df = df[securities].dropna()
    X1 = np.array(temp_df[securities[0]])
    X2 = np.array(temp_df[securities[1]])
    X3 = np.array(temp_df[securities[2]])
    X = np.array((temp_df[securities]-temp_df.shift(1)[securities]).dropna())
    pca_model = PCA()
    pca_model.fit(X)
    weights = pca_model.components_[2,:]
    #normalize the weights
    weights = weights/weights[1]*2.
    print weights
    #calc the fly
    return weights[0]*df[securities[0]]+weights[1]*df[securities[1]]+weights[2]*df[securities[2]]
    pass



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

#Start Analysis
filename = r'/Users/Abhishek/Desktop/Projects/Bond-Swap-Fly-monitor/Datafin.xlsx'
test_frame = gather_data(filename,'Data-Bonds')
ret_frame = process_headers(test_frame)
ret_df = process_timeline(ret_frame)

#Get headers - for this just create a sheet called Sheet 31 (or anything else literally and add the input table)

filename_headers = r'/Users/Abhishek/Desktop/Projects/Bond-Swap-Fly-monitor/Datafin.xlsx'
headers = pd.read_excel(filename_headers,sheetname='Sheet 31')
headers.index = headers.iloc[:,0]
headers = headers.iloc[:,1:]
print headers.head()

#Plot time series of a given fly for all other countries

fly_list = [2, 5, 10]
import matplotlib.pyplot as plt

plt.style.use('ggplot')
fly_country_dict = {key: np.nan for key in list(headers.columns)}
pca_fly_country_dict = {key: np.nan for key in list(headers.columns)}
for country in fly_country_dict.keys():
    tickerlist = [headers.loc[str(x) + 'Y', country] for x in fly_list]
    metric = 'YLD_YTM_MID'
    tickerlist = [x + ' Govt__' + metric for x in tickerlist]
    print country
    print tickerlist

    try:
        pca_fly_country_dict[country] = pd.DataFrame(fly_pca_calc(ret_df, tickerlist).dropna(),
                                                     columns=['PCA_Fly' + ''.join([str(x) for x in fly_list])])
        fly_country_dict[country] = pd.DataFrame(fly_calc(ret_df, tickerlist).dropna(),
                                                 columns=['1-2-1_Fly' + ''.join([str(x) for x in fly_list])])
        ax = pca_fly_country_dict[country].plot(legend=True)
        fly_country_dict[country].plot(legend=True, ax=ax)

        # axes = sns.tsplot(pca_fly_country_dict[country].dropna())
        # axes1 = sns.tsplot(fly_country_dict[country].dropna())
        # plt.legend([axes.plot,axes1.plot],[country+'pca',country+'1-2-1'])
    except:
        print "Did not work"
    plt.show()



#For Sheet - Data-SwapRates
pca_fly_us_2510 = fly_pca_calc(ret_df,map(list(ret_df.columns).__getitem__,[2,4,6]))
fly_list = ['BPSW3 Curncy','BPSW5 Curncy','BPSW7 Curncy']; fly_list = [x+'__PX_LAST' for x in fly_list]
fly_index = [list(ret_df.columns).index(x) for x in fly_list]
pca_fly_bp_2510 = fly_pca_calc(ret_df,map(list(ret_df.columns).__getitem__,fly_index))


#Start of Part 1

# Intra-Country flies regressions for both PCA and 1-2-1 weighted flies
import statsmodels.regression.linear_model as sm
import sklearn.metrics as sk_m


def realign_dates(df_list):
    df_list = [pd.DataFrame(x) for x in df_list]
    master_df = df_list[0]
    for x in df_list[1:]:
        master_df = master_df.merge(x, how='outer', left_index=True, right_index=True)

    master_df = master_df.dropna()
    df_list = [master_df[col] for col in master_df.columns]
    return df_list


def ret_regression(ret_df, Ylist, Y_country, Xlist, X_country, metric='YLD_YTM_MID', del_changes=True):
    Y_tickerlist = [headers.loc[str(x) + 'Y', Y_country] + ' Govt__' + metric for x in Ylist]
    X_tickerlist = [headers.loc[str(x) + 'Y', X_country] + ' Govt__' + metric for x in Xlist]

    Y_country = Y_country + ''.join([str(x) for x in Ylist])
    X_country = X_country + ''.join([str(x) for x in Xlist])

    if Y_country == X_country:
        print "Regression Not Possible"
        return 0.

    fly_country_dict = {key: np.nan for key in [Y_country, X_country]}
    pca_fly_country_dict = {key: np.nan for key in [Y_country, X_country]}

    try:
        print Y_country + ' PCA_Fly Weights'
        pca_fly_country_dict[Y_country] = pd.DataFrame(fly_pca_calc(ret_df, Y_tickerlist).dropna()
                                                       , columns=[Y_country + 'PCA_Fly'])
        print X_country + ' PCA_Fly Weights'
        pca_fly_country_dict[X_country] = pd.DataFrame(fly_pca_calc(ret_df, X_tickerlist).dropna()
                                                       , columns=[X_country + 'PCA_Fly'])

        fly_country_dict[Y_country] = pd.DataFrame(fly_calc(ret_df, Y_tickerlist).dropna()
                                                   , columns=[Y_country + '1-2-1_Fly'])
        fly_country_dict[X_country] = pd.DataFrame(fly_calc(ret_df, X_tickerlist).dropna()
                                                   , columns=[X_country + '1-2-1_Fly'])

    except:
        print "Did not work at data referencing level"
        return 0.
    if del_changes:
        lm = sk.linear_model.LinearRegression(fit_intercept=False)
        lm_1 = sk.linear_model.LinearRegression(fit_intercept=False)
        # We want to fit the model Del Fly1 ~ Del Fly2

        Y = pca_fly_country_dict[Y_country] - pca_fly_country_dict[Y_country].shift(1)
        Y = Y.dropna()
        X = pca_fly_country_dict[X_country] - pca_fly_country_dict[X_country].shift(1)
        X = X.dropna()
        input_list = realign_dates([Y, X])
        Y = np.array(input_list[0]).reshape((-1, 1));
        X = np.array(input_list[1]).reshape((-1, 1))

        # model = sm.OLS(Y,X,hasconst = True)
        # results = model.fit()
        lm.fit(X=X, y=Y)

        Y_1 = fly_country_dict[Y_country] - fly_country_dict[Y_country].shift(1)
        Y_1 = Y_1.dropna()
        X_1 = fly_country_dict[X_country] - fly_country_dict[X_country].shift(1)
        X_1 = X_1.dropna()
        input_list = realign_dates([Y_1, X_1])
        Y_1 = np.array(input_list[0]).reshape((-1, 1));
        X_1 = np.array(input_list[1]).reshape((-1, 1))
        lm_1.fit(X=X_1, y=Y_1)
        # return results
        return {'PCA': {'beta': lm.coef_[0][0], 'r2-score': sk_m.r2_score(Y, lm.predict(X))},
                '1-2-1': {'beta': lm_1.coef_[0][0], 'r2-score': sk_m.r2_score(Y_1, lm_1.predict(X_1))}}
    else:
        lm = sk.linear_model.LinearRegression()
        Y = pca_fly_country_dict[Y_country]
        X = pca_fly_country_dict[X_country]
        input_list = realign_dates([Y, X])
        Y = np.array(input_list[0]).reshape((-1, 1));
        X = np.array(input_list[1]).reshape((-1, 1))
        lm.fit(X=X, y=Y)
        return {'beta': lm.coef_[0][0], 'intercept': lm.intercept_[0][0], 'r2-score': sk_m.r2_score(Y, lm.predict(X))}


print 'Check 2-5-10 vs 3-5-7 as both are medium tenors'

Ylist = [2,5,10]
Xlist = [3,5,7]
for country in list(headers.columns):
    print 'Regression Results'
    print ret_regression(ret_df,Ylist,country,Xlist,country)


print r'Check 2-3-5 vs 5-10-30 to compare the convexity relationships across curves'

Ylist = [2,3,5]
Xlist = [5,10,30]
for country in list(headers.columns):
    print 'Regression Results'
    print ret_regression(ret_df,Ylist,country,Xlist,country)
# As expected, very little correlation between the short end and the lond end convexity

#Check 5-7-10 vs 2-10-30 to compare movement of belly convexity and the curve convexity

Ylist = [5,7,10]
Xlist = [2,10,30]
for country in list(headers.columns):
    print 'Regression Results'
    print ret_regression(ret_df,Ylist,country,Xlist,country)


#Inter-Country flies regression
#France vs Germany
X_country = 'GER'
Y_country = 'FRA'
compare_fly_list = [[2,3,5],[3,5,7],[5,7,10],[5,10,30]]
for fly in compare_fly_list:
    print 'Regression Results - ' +','.join([str(x) for x in fly])
    print ret_regression(ret_df,fly,Y_country,fly,X_country)

