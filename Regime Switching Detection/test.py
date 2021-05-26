from __future__ import print_function
import datetime
import pickle
import warnings
import datetime
from hmmlearn.hmm import GaussianHMM
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
import myfunc as mf


def read_data():
    # "BTC-USD": Bitcoin, "ETH-USD": Etherium
    print("Read data")
    df = pd.read_excel('Dataset_loader.xlsx', index_col=0)
    US10YF_USDJPY = pd.read_excel('US future JPY dollar.xlsx', index_col=0)
    SP1 = pd.read_excel('SP1 Index.xlsx', index_col=0)
    Gold_VIX = pd.read_excel('Gold VIX Futures.xlsx', index_col=0)
    new_df = pd.concat([df, US10YF_USDJPY, SP1, Gold_VIX], axis=1)
    return new_df

def process_data():
    df = read_data()
    print("Start processing data")

    if "SPY" in df.columns:
        df["SPY Return"]=df["SPY"].pct_change()
    else:
        print("the name of the column does not exist")
        exit()
    
    if "VIX" in df.columns:
        df["VIX Return"]=df["VIX"].pct_change()
    else:
        print("the name of the column does not exist")
        exit()

    if "Gold" in df.columns:
        df["Gold Return"]=df["Gold"].pct_change()
    else:
        print("the name of the column does not exist")
        exit()
    
    if "USDJPY" in df.columns:
        df["USDJPY Return"]=df["USDJPY"].pct_change()
    else:
        print("the name of the column does not exist")
        exit()

    if "US10YT-F" in df.columns:
        df["US10YT-F Return"]=df["US10YT-F"].pct_change()
    else:
        print("the name of the column does not exist")
        exit()

    if "FEDL01 Index" in df.columns:
        df["FEDL Diff"]=df["FEDL01 Index"].diff()
    else:
        print("the name of the column does not exist")
        exit()
    
    matches = ["GT10 Govt", "GT2 Govt"]
    if any(x in df.columns for x in matches):
        df['YC Deriv']=df['GT10 Govt']-df['GT2 Govt']
        df['GT10 Diff']=df['GT10 Govt'].diff()
    else:
        print("the name of the column does not exist")
        exit()
    
    if "CPI YOY Index" in df.columns:
        df['CPI Diff']=df['CPI YOY Index'].diff()
    else:
        print("the name of the column does not exist")
        exit()
    
    if "LBUTTRUU Index" in df.columns:
        df['LBUTTRUU Return']=df['LBUTTRUU Index'].pct_change()
    else:
        print("the name of the column does not exist")
        exit()

    if "XAU Curncy" in df.columns:
        df['XAU Return']=df['XAU Curncy'].pct_change()
    else:
        print("the name of the column does not exist")
        exit()

    print("Finish processing data")
    return df

def standardscaler_df(df):
    print("Standard scaler for whole dataframe is running")
    df_new = df
    min_max_scaler = MinMaxScaler()
    df_new[df_new.columns]=min_max_scaler.fit_transform(df_new[df_new.columns])
    return df_new

def bear_bull_detection_rolling(df,object="Return",ew = 30): 
    # Rolling window / df : dataframe , ew: in sample (datetime)
    print("Start bear_bull_detection_rolling")

    df.dropna(inplace=True)
    df_final = df.copy()
    df_scaled = standardscaler_df(df)
    bb_hidden_states_daily = np.array([])
    in_sample_length= ew

    for i in range(len(df_scaled[in_sample_length:])):
        fit_array_rets = np.column_stack([df_scaled[object][i:in_sample_length+i]])
        hmm_model = GaussianHMM(n_components=2, covariance_type="full", n_iter=1000).fit(fit_array_rets)
        # learn through in-sample
        hs_array = hmm_model.predict(np.column_stack([df_scaled[object][in_sample_length+i]]))
        # what is current state?
        a = hmm_model.means_[0]
        b = hmm_model.means_[1]

        if (a > b) & (hs_array == 0):
            bb_hidden_states_daily = np.append(bb_hidden_states_daily,"bull")
        elif (a < b) & (hs_array == 0):
            bb_hidden_states_daily = np.append(bb_hidden_states_daily,"bear")
        elif (a > b) & (hs_array == 1):
            bb_hidden_states_daily = np.append(bb_hidden_states_daily,"bear")
        else :
            bb_hidden_states_daily = np.append(bb_hidden_states_daily,"bull")
    
    print("Bear Bull Detection complete!!")
    
    bb = pd.DataFrame(bb_hidden_states_daily,index=df_scaled[in_sample_length:].index)
    df_final["Bear Bull"]=bb
    df_final.dropna(inplace=True)
    return df_final

def bear_bull_detection_expanding(df,object="Return",ew=datetime.datetime(2012,4,1)): 
    # Expanding window / df : dataframe , ew: in sample (datetime)
    print("Start bear_bull_detection_expanding")

    df.dropna(inplace=True)
    df_final = df.copy()
    df_scaled = standardscaler_df(df)
    bb_hidden_states_daily = np.array([])
    in_sample_length=len(df_scaled[df_scaled.index < ew])

    #print(in_sample_length)
    for i in range(len(df_scaled[in_sample_length:])):
        fit_array_rets = np.column_stack([df_scaled[:in_sample_length+1+i][object]])
        hmm_model = GaussianHMM(n_components=2, covariance_type="full", n_iter=1000).fit(fit_array_rets)
        hs_array = hmm_model.predict(np.column_stack([df_scaled[:in_sample_length+1+i][object]]))
        a = hmm_model.means_[0]
        b = hmm_model.means_[1]

        if (a > b) & (hs_array[-1]== 0):
            bb_hidden_states_daily = np.append(bb_hidden_states_daily,"bull")
        elif (a < b) & (hs_array[-1]== 0):
            bb_hidden_states_daily = np.append(bb_hidden_states_daily,"bear")
        elif (a > b) & (hs_array[-1]== 1):
            bb_hidden_states_daily = np.append(bb_hidden_states_daily,"bear")
        else :
            bb_hidden_states_daily = np.append(bb_hidden_states_daily,"bull")
        print("bb_hidden_states_daily:",bb_hidden_states_daily)

    print("Bear Bull Detection complete!!")

    bb = pd.DataFrame(bb_hidden_states_daily,index=df_scaled[in_sample_length:].index)    
    df_final["Bear Bull"]=bb
    df_final.dropna(inplace=True)
    return df_final

def bear_bull_detection_MI_expanding(df,object,object2="US10YT-F 1d Return",ew=datetime.datetime(2012,4,1)): 
    # Expanding window / df : dataframe , ew: in sample (datetime)
    print("Start bear_bull_detection_MI_expanding")

    df.dropna(inplace=True)
    df_final = df.copy()
    df_scaled = standardscaler_df(df)
    bb_hidden_states_daily = np.array([])
    in_sample_length=len(df_scaled[df_scaled.index < ew])

    #print(in_sample_length)
    for i in range(len(df_scaled[in_sample_length:])):
        fit_array_rets = np.column_stack([df_scaled[object][:in_sample_length+1+i]])
        hmm_model = GaussianHMM(n_components=2, covariance_type="full", n_iter=1000).fit(fit_array_rets)
        hs_array = hmm_model.predict(np.column_stack([df_scaled[object2][:in_sample_length+1+i]]))
        #print("hs_array[-1]:",hs_array[-1])
        a = hmm_model.means_[0][object.index(object2)]
        b = hmm_model.means_[1][object.index(object2)]
        #print("a:",a)
        #print("b:",b)
        if (a > b) & (hs_array[-1]== 0):
            bb_hidden_states_daily = np.append(bb_hidden_states_daily,"bull")
        elif (a < b) & (hs_array[-1]== 0):
            bb_hidden_states_daily = np.append(bb_hidden_states_daily,"bear")
        elif (a > b) & (hs_array[-1]== 1):
            bb_hidden_states_daily = np.append(bb_hidden_states_daily,"bear")
        else :
            bb_hidden_states_daily = np.append(bb_hidden_states_daily,"bull")
        print("bb_hidden_states_daily:",bb_hidden_states_daily)
        
    print("Bear Bull Detection complete!!")

    bb = pd.DataFrame(bb_hidden_states_daily,index=df_scaled[in_sample_length:].index)    
    df_final["Bear Bull"]=bb
    df_final.dropna(inplace=True)
    return df_final

def bear_bull_plot(df,object_1="Return",object_2="Close",object_3="SPY"):
    print("plot_bull_bear")

    df["Bear Bull"] = df["Bear Bull"].shift(1)

    df.dropna(inplace=True)

    fig, axs = plt.subplots(2,3, figsize=(30,8),sharex='col',sharey='col')
    bear_bull = 'Bear Bull'
    axs[0,0].set_title("Bear")
    axs[0,1].set_title("Bear")
    axs[1,0].set_title("Bull")
    axs[1,1].set_title("Bull")
    axs[0,2].set_title("Bull")
    axs[1,2].set_title("Bear")
    axs[0,2].set(ylabel='SPY Index')
    axs[1,2].set(ylabel='SPY Index')
    axs[0,0].hist(df[object_1][df[bear_bull] == "bear"],bins=100, color = cm.rainbow(0.))
    axs[1,0].hist(df[object_1][df[bear_bull] == "bull"],bins=100, color = cm.rainbow(1.))
    axs[0,1].plot_date(df.index[df[bear_bull] == "bear"],df[object_2][df[bear_bull] == "bear"],color = cm.rainbow(0.),markersize=2)
    axs[1,1].plot_date(df.index[df[bear_bull] == "bull"],df[object_2][df[bear_bull] == "bull"],color = cm.rainbow(1.),markersize=2)
    axs[0,2].plot_date(df.index[df[bear_bull] == "bear"],df[object_3][df[bear_bull] == "bear"],color = cm.rainbow(0.),markersize=2)
    axs[1,2].plot_date(df.index[df[bear_bull] == "bull"],df[object_3][df[bear_bull] == "bull"],color = cm.rainbow(1.),markersize=2)
    axs[0,0].grid(True)
    axs[0,1].grid(True)
    axs[1,0].grid(True)
    axs[1,1].grid(True)
    axs[0,2].grid(True)
    axs[1,2].grid(True)
    plt.show()

def main():
    # get dataframe and pre-process data
    df = process_data()
    # Post-processing for 5-day return & scaling
    df_new = df
    # US10YT-F Daily return
    df_new["US10YT-F 1d Return"]=df_new["US10YT-F"].pct_change()
    df_new.dropna(inplace=True)
    
    ########################################################

    # Regime detection
    df_after_regime_detection_MI_expanding = bear_bull_detection_MI_expanding(df_new,["SPY Return","VIX"],"SPY Return",datetime.datetime(2014,4,1))
    #df_after_regime_detection_expanding = bear_bull_detection_expanding(df_new,"SPY Return",datetime.datetime(2014,4,1))
    df_after_regime_detection_MI_expanding.to_csv('df_after_detection_MI_expanding.csv')

    df_after_regime_detection_MI_expanding =pd.read_csv('df_after_detection_MI_expanding.csv',index_col=0,parse_dates=True)
    #print(df_after_regime_detection_expanding)

    bear_bull_plot(df_after_regime_detection_MI_expanding, "SPY Return","US10YT-F","SPY")


if __name__ == "__main__":
    main()

