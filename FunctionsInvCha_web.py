# Libraries needed
import numpy as np
import pandas as pd
import pandas_datareader as web 
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score
import xgboost as xgb
import pickle
import pandas.util.testing as tm

# FUNCTIONS NEEDED
######################## CREATE BASIC DATAFRAME TO FIT THE ML MODELS
def create_data(ticker):
    """
    Download data from yahoo, compute the up/down signal and technical indicators needed, 
    drop columns that are not needed.
    Data time period and other signals could be added.
    INPUT: ticker = the ticker of the stocks you want to use.
    OUTPUT: dataframe with the y variable (direction) and the features needed.
    """
    # Download data and compute
    data = web.DataReader(ticker, "yahoo", "2000-01-01", "2021-03-01") 
    sp = web.DataReader("SPY", "yahoo", "2000-01-01", "2021-03-01")
    sp["Ret_SP"] = sp["Adj Close"].pct_change()
    sp["UD_SP"] = np.where(sp["Ret_SP"].shift(-1) > 0, 1, 0)
    sp["10days_Vol"] = sp["Ret_SP"].rolling(10).std()
    sp = sp.dropna()
    data["Ret"] = data["Adj Close"].pct_change()
    data["Signal"] = np.where(data["Ret"].shift(-1) > 0, 1, 0)
    data = data.dropna()
    # Technical indicators
    #RSI
    data["RSI"] = ta.momentum.rsi(close = data["Close"], n=2)
    #Stochastic Oscillator (I will use the signal as an input)
    data["SR_signal"] = ta.momentum.stoch_signal(close=data["Close"], 
                                                high=data["High"], low=data["Low"], n=5)
    #MACD (I will use the signal as an input, with default values for the two MA)
    data["MACD"] = ta.trend.macd_signal(close=data["Close"])
    #Average true range (ATR) for 5-days period
    data["ATR"] = ta.volatility.average_true_range(high=data["High"], 
                                                low=data["Low"], close=data["Close"], n=5)
    
    sp = sp.drop(columns=["Open", "High", "Low", "Close", "Volume", "Adj Close", "Ret_SP"])
    data = data.merge(sp, on="Date")

    touse = data.drop(columns=["Open", "High", "Low", "Close", "Volume", "Adj Close",  "Ret"])
    touse = touse.dropna()
    return touse

################### Same as above but used to store data in csv and new download each time
def download_and_save_data(ticker, start, end):
    """
    Same as create_data but used to store data downloaded in CSV file if needed.
    """
    data = web.DataReader(ticker, "yahoo", start=start, end=end) 
    sp = web.DataReader("SPY", "yahoo", start=start, end=end)
    sp["Ret_SP"] = sp["Adj Close"].pct_change()
    sp["UD_SP"] = np.where(sp["Ret_SP"].shift(-1) > 0, 1, 0)
    sp["10days_Vol"] = sp["Ret_SP"].rolling(10).std()
    sp = sp.dropna()
    data["Ret"] = data["Adj Close"].pct_change()
    data["Signal"] = np.where(data["Ret"].shift(-1) > 0, 1, 0)
    data = data.dropna()
    data["RSI"] = ta.momentum.rsi(close = data["Close"], n=2)
    data["SR_signal"] = ta.momentum.stoch_signal(close=data["Close"], 
                                                high=data["High"], low=data["Low"], n=5)
    data["MACD"] = ta.trend.macd_signal(close=data["Close"])
    data["ATR"] = ta.volatility.average_true_range(high=data["High"], 
                                                low=data["Low"], close=data["Close"], n=5)

    sp = sp.drop(columns=["Open", "High", "Low", "Close", "Volume", "Adj Close", "Ret_SP"])
    data = data.merge(sp, on="Date")
    touse = data.drop(columns=["Volume", "Ret"])   # store also H-L-O-C
    touse = touse.dropna()
    touse.to_csv("/Users/AlessioGiust/Documents/Ale/Python - R/Finance PY/Investment challenge/downloaded_data/{}.csv".format(ticker))

################### CREATE FUNCTION TO SPLIT DATA NOT RANDOMLY
def split_data(df, inputs, output, pct_in, drop=False):
    """ 
    Create non random train_test_split.
    INPUTS:
    - df = dataframe to use
    - inputs = list of input columns in the df
    - output = name of output column
    - pct_in = percentage of datapoints to include in the training sample
    - drop = if True use the drop function to keep all the columns except the one indicated in the input parameter 
    OUTPUTS: X_train, X_test, y_train, y_test.
    """
    if drop == False:
        X = np.array(df[inputs])
        y = np.array(df[output])   
        split = int(len(df)*pct_in)
        X_train = X[0:split]  
        X_test = X[split:-1] #use:-1, it drops the last row because we don't have the label value for the next day
        y_train = y[0:split]
        y_test = y[split:-1]
        return X_train, X_test, y_train, y_test
    else:
        X = np.array(df.drop(columns=inputs))
        y = np.array(df[output])   
        split = int(len(df)*pct_in)
        X_train = X[0:split]  
        X_test = X[split:-1]
        y_train = y[0:split]
        y_test = y[split:-1]
        return X_train, X_test, y_train, y_test, #X, y

################## FUNCTIONS TO FIT THE MODELS and EVALUATE ACCURACY ###################
####### RANDOM FOREST ##################################################################
def random_forest_model(data, inputs, output, pct_in, n_trees, all_data=False):
    """ 
    Split the data, fit a tree and compute accuracy score (hit ratio).
    INPUTS:
    - data = dataframe for the stock obtained with the create_data function
    - inputs = list of input columns in the df
    - output = name of output column
    - pct_in = percentage of datapoints to include in the training sample 
    n_trees = number of trees to use in the random forest 
    OUTPUTS: trained model, accuracy and features importance.
    """
    if all_data == False:
        # Split the data 
        X_train, X_test, y_train, y_test = split_data(df=data, inputs=inputs, output=output, pct_in=pct_in)
        # Fit the model
        rf = RandomForestClassifier(n_estimators=n_trees, oob_score=True, criterion="gini", random_state=0)
        modelrf = rf.fit(X_train, y_train)
        # Obtain accuracy
        y_pred = modelrf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        feat_imp = pd.Series(modelrf.feature_importances_, index=['RSI', 'SR_signal', 'MACD', 'ATR', 'UD_SP', '10days_Vol'])
        return modelrf, acc, feat_imp
    else:
        # use all the data available to fit the model
        rf = RandomForestClassifier(n_estimators=n_trees, oob_score=True, criterion="gini", random_state=0)
        X = np.array(data[inputs])
        y = np.array(data[output])
        modelrf = rf.fit(X, y)
        return modelrf


####### XGBOOST ########################################################################
def xgboost_model(data, inputs, output, pct_in, max_depth, min_child_weight, all_data=False):
    """ 
    Split the data, fit xgboost model and compute accuracy score.
    INPUTS:
    - data = dataframe for the stock obtained with the create_data function
    - inputs = list of input columns in the df
    - output = name of output column
    - pct_in = percentage of datapoints to include in the training sample 
    OUTPUTS: trained model, accuracy and features importance.
    """
    if all_data == False:
        X_train, X_test, y_train, y_test = split_data(df=data, inputs=inputs, output=output, pct_in=pct_in)
        xgb_classifier = xgb.XGBClassifier(max_depth=max_depth, min_child_weight=min_child_weight)
        modelXG = xgb_classifier.fit(X_train, y_train)
        predictions = modelXG.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        feat_imp = pd.Series(modelXG.feature_importances_, index=['RSI', 'SR_signal', 'MACD', 'ATR', 'UD_SP', '10days_Vol'])
        return modelXG, acc, feat_imp
    else:
        xgb_classifier = xgb.XGBClassifier(max_depth=max_depth, min_child_weight=min_child_weight)
        X = np.array(data[inputs])
        y = np.array(data[output])
        modelXG = xgb_classifier.fit(X, y)
        return modelXG


def tune_param_XGB(param1_list, param2_list, data, inputs, output, pct_in):
    """Function to return the accuracy and the value of the parameters max_depth and min_child_weight that maximizes it, all other parameter kept constant.
    - param1_list = list of max_depth parameters to test 
    - param2_list = list of min_child_weight parameters to test 
    - other parameters are the train and test values for X and y to use in the model"""
    X_train, X_test, y_train, y_test = split_data(df=data, inputs=inputs,
    output=output, pct_in=pct_in)
    res = []    # store results
    for p1 in param1_list:
        for p2 in param2_list:
            xgb_classifier = xgb.XGBClassifier(max_depth=p1, min_child_weight=p2)
            modelXG = xgb_classifier.fit(X_train, y_train)
            predictions = modelXG.predict(X_test)
            acc = accuracy_score(y_test, predictions)
            res.append([acc, p1, p2])
    return max(res)   #Optimal parameters based on higest accuracy  
    
####### LINEAR DISCRIMINANT ANALYSIS #####################################################
def LDA_class(data, inputs, output, pct_in, all_data=False):
    """
    Function to train LDA Classification model given data.
    INPUTS: same as XGB function.
    OUTPUTS: trained model and accuracy.
    """ 
    if all_data == False:
        X_train, X_test, y_train, y_test = split_data(df=data, inputs=inputs, output=output, pct_in=pct_in)
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train, y_train)
        y_p = lda.predict(X_test)
        acc = accuracy_score(y_test, y_p)
        return lda, acc
    else:
        lda = LinearDiscriminantAnalysis()
        X = np.array(data[inputs])
        y = np.array(data[output])
        lda.fit(X, y)
        return lda

####### QUADRATIC DISCRIMINANT ANALYSIS ###################################################
def QDA_class(data, inputs, output, pct_in, all_data=False):
    """
    Function to train QDA Classification model given data.
    INPUTS: same as XGB function.
    OUTPUTS: trained model and accuracy.
    """
    if all_data == False:
        X_train, X_test, y_train, y_test = split_data(df=data, inputs=inputs, output=output, pct_in=pct_in)
        qda = QuadraticDiscriminantAnalysis()
        qda.fit(X_train, y_train)
        y_p = qda.predict(X_test)
        acc = accuracy_score(y_test, y_p)
        return qda, acc
    else:
        qda = QuadraticDiscriminantAnalysis()
        X = np.array(data[inputs])
        y = np.array(data[output])
        qda.fit(X, y)
        return qda


##########################################################################################
##########################################################################################

############# GET DATA UP TO TODAY, COMPUTE INDICATORS
def compute_data_toadd(ticker):  #date
    """
    Download data from yahoo, compute the up/down signal and technical indicators needed to predict, 
    drop columns that are not needed.
    Data time period and other signals could be added.
    INPUT: ticker = the ticker of the stocks you want to use.
    OUTPUT: entire dataframe and features needed today to predict for tomorrow.
    """
    from datetime import date
    try:
        data = web.DataReader(ticker, "yahoo", "2021-02-01", date.today())
        data = data.dropna()
        sp = web.DataReader("SPY", "yahoo", "2021-02-01", date.today())
        sp["Ret_SP"] = sp["Adj Close"].pct_change()
        sp["UD_SP"] = np.where(sp["Ret_SP"]>0, 1, 0)
        sp["10days_Vol"] = sp["Ret_SP"].rolling(10).std()
        sp = sp.dropna()

        #RSI
        data["RSI"] = ta.momentum.rsi(close = data["Close"], n=2)
        #Stochastic Oscillator (I will use the signal as an input)
        data["SR_signal"] = ta.momentum.stoch_signal(close=data["Close"], 
                                                    high=data["High"], low=data["Low"], n=5)
        #MACD (I will use the signal as an input, with default values for the two MA)
        data["MACD"] = ta.trend.macd_signal(close=data["Close"])
        #Average true range (ATR) for 5-days period
        data["ATR"] = ta.volatility.average_true_range(high=data["High"], 
                                                    low=data["Low"], close=data["Close"], n=5)
        
        data = data.merge(sp[["UD_SP", "10days_Vol"]], on="Date")

        return data, data[["RSI", "SR_signal", "MACD", "ATR", "UD_SP", "10days_Vol"]].loc[date.today().strftime("%Y-%m-%d")]
    except KeyError:
        print("The market is closed today")

############ SCREENING ESG STOCKS TO SEE FOR WHICH WE HAVE HIGHEST PREDICTION ACCURACY
def screen_stocks(tickers, inputs, output, pct_in):
    """
    Create a dataframe to screen stocks based on the average accuracy from OOS testing.
    INPUTS: same as XGB and other ML models functions.
    OUTPUTS: dataframe with top 25 stocks sorted by accuracy.
    """
    storage = pd.DataFrame(columns=["Ticker", "Avg Acc"])
    for tick in tickers:
        try:
            df = create_data(tick)
            accrf = random_forest_model(df, inputs=inputs, output=output, pct_in=pct_in,n_trees=100)[1] 
            accxgb = xgboost_model(df, inputs=inputs, output=output, pct_in=pct_in, max_depth=6, min_child_weight=1)[1]
            accavg = (accrf + accxgb)/2
            storage = storage.append({"Ticker": tick, "Avg Acc": accavg}, ignore_index=True)
        except KeyError:
            pass

    return storage.sort_values("Avg Acc", ascending=False).head(25) #sort from the highest, take first 25

############ STORE FITTED MODEL
def store_fitted_models(tickers, inputs, output, pct_in):
    for tick in tickers:
        df_to_fit = create_data(tick)
        # Tune hyper
        tuned = tune_param_XGB([2,3,4,5,6,7,8,9,10], [1,2,3,4,5,6,7,8,9,10], data=df_to_fit, inputs=inputs, output=output, pct_in=pct_in)
        # Fit models
        mod_forest = random_forest_model(df_to_fit, inputs=inputs, output=output, pct_in=pct_in, n_trees=100, all_data=True)
        mod_xgb = xgboost_model(df_to_fit, inputs=inputs, output=output, pct_in=pct_in, max_depth=tuned[1], min_child_weight=tuned[2], all_data=True)
        mod_lda = LDA_class(df_to_fit, inputs=inputs, output=output, pct_in=pct_in, all_data=True)
        mod_qda = QDA_class(df_to_fit, inputs=inputs, output=output, pct_in=pct_in, all_data=True)
        # Store models with pickle
        pickle.dump(mod_forest, open(f"modelsML/modelRF_{tick}.pkl", "wb")) 
        pickle.dump(mod_xgb, open(f"modelsML/modelXGB_{tick}.pkl", "wb")) 
        pickle.dump(mod_lda, open(f"modelsML/modelLDA_{tick}.pkl", "wb")) 
        pickle.dump(mod_qda, open(f"modelsML/modelQDA_{tick}.pkl", "wb")) 


############ COMBINE THE FUNCTIONS DEFINED ABOVE TO PREDICT PRICE DIRECTION FOR TOMORROW
def model_application(tickers):  
    """
    Used in the Jupyter Nootebook to get predictions for all stocks.
    1) Load fitted models
    2) Obtain data and indicators for today (yahoo data in real time, small delay) 
    3) Use the fitted model to predict the direction for next day
    INPUTS: stock tickers
    OUTPUTS: predictions for each ML model for tomorrow.
    """
    for tick in tickers:
        #1) Load and use with pickle
        mod_forest = pickle.load(open(f"modelsML/modelRF_{tick}.pkl", "rb"))
        mod_xgb = pickle.load(open(f"modelsML/modelXGB_{tick}.pkl", "rb"))
        mod_lda = pickle.load(open(f"modelsML/modelLDA_{tick}.pkl", "rb"))
        mod_qda = pickle.load(open(f"modelsML/modelQDA_{tick}.pkl", "rb"))
        # 2) function compute_data_toadd
        data_needed = compute_data_toadd(ticker=tick)[1]
        X_to_predict = np.array(data_needed).reshape(1, -1)
        # 3) Predict and print results
        final_output_rf = mod_forest.predict(X_to_predict)
        final_output_xg = mod_xgb.predict(X_to_predict)
        final_output_lda = mod_lda.predict(X_to_predict)
        final_output_qda = mod_qda.predict(X_to_predict)
        print("----------------------------------------")
        print(f"--------- Predictions for {tick} ---------")
        print("----------------------------------------")
        print("Random Forest: ", "UP" if final_output_rf==1 else "DOWN")
        print("XGBoost:       ", "UP" if final_output_xg==1 else "DOWN")
        print("LDA:           ", "UP" if final_output_lda==1 else "DOWN")
        print("QDA:           ", "UP" if final_output_qda==1 else "DOWN")
        print("----------------------------------------")

############ MODEL APPLICATION for WEBSITE
def model_application_for_web(tick):
    """
    Uses a single tick (stock) that can be selected by the user, used in the stremlit app.
    """
    # Load and use with pickle
    mod_forest = pickle.load(open(f"modelsML/modelRF_{tick}.pkl", "rb"))
    mod_xgb = pickle.load(open(f"modelsML/modelXGB_{tick}.pkl", "rb"))
    mod_lda = pickle.load(open(f"modelsML/modelLDA_{tick}.pkl", "rb"))
    mod_qda = pickle.load(open(f"modelsML/modelQDA_{tick}.pkl", "rb"))
    # function compute_data_toadd
    data_needed = compute_data_toadd(ticker=tick)[1] 
    X_to_predict = np.array(data_needed).reshape(1, -1)
    # predict
    final_output_rf = mod_forest.predict(X_to_predict)
    final_output_xg = mod_xgb.predict(X_to_predict)
    final_output_lda = mod_lda.predict(X_to_predict)
    final_output_qda = mod_qda.predict(X_to_predict)
    # Output for the website
    if final_output_rf==1:
        rf = "Random Forest prediction: The stock price will go UP"
    else: 
        rf = "Random Forest prediction: The stock price will go DOWN"
    if final_output_xg==1:
        xg = "XGBoost prediction: The stock price will go UP"
    else: 
        xg = "XGBoost prediction: The stock price will go DOWN"
    if final_output_lda==1:
        lda = "LDA prediction: The stock price will go UP"
    else: 
        lda = "LDA prediction: The stock price will go DOWN"
    if final_output_qda==1:
        qda = "QDA prediction: The stock price will go UP"
    else: 
        qda = "QDA prediction: The stock price will go DOWN"
    return rf, xg, lda, qda