import pandas as pd
import numpy as np
import scipy.stats as sp
import statsmodels.api as sm
import matplotlib.pyplot as plt
import arch as ac
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import ttest_1samp, jarque_bera

##### Further possible additions:                       
# VaR quantile regression 
# VaR with extreme value theory (no good library for EVT in python)
# VaR and ES with Monte Carlo simulation (GARCH, GBM, JUMP, FACTORIAL)


def VaR_basic(r, alpha=0.01, h=1, dist="gaussian", position=1.0, df=5, rolling=False, start=60, win=252, garch=False):
    """
    Compute VaR based on GAUSSIAN, TSTUDENT, HISTORICAL sim., CORNER-FISHER approx. and volatility predicted by a GRACH(1,1) model.
    Rule of the square root of h to obtain the VaR at h days, or just pass h-days returns as input.
    - r = pandas series of returns
    - start = minimum number of datapoints to use in the rolling function
    - win = number of datapoints in the rolling window
    - garch = boolean, if True the volatility used to compute the VaR is a GARCH(1,1) forecast obtained using the function garch11_vol_fixedspec
    OUTPUT = single value for the VaR 
    """
    if rolling == False: 
        mean = r.mean()*h
        std = r.std()*np.sqrt(h)
        quant = np.quantile(r, alpha)
        if garch == False:
            if dist == "gaussian": VAR = -position*(mean+sp.norm.ppf(alpha)*std)
            elif dist == "tstudent": VAR = -position*(mean+sp.t.ppf(alpha, df)*std)
            elif dist == "hist": VAR = (-position*quant)*np.sqrt(h)   
            else: raise ValueError("Please use an accepted distribution type")
        else:
            vol = garch11_vol_fixedspec(r, start)
            vol = float(vol.iloc[-1])
            if dist == "gaussian": VAR = -position*(mean+sp.norm.ppf(alpha)*vol)
            elif dist == "tstudent": VAR = -position*(mean+sp.t.ppf(alpha, df)*vol)
            elif dist == "hist": VAR = (-position*quant)*np.sqrt(h)  # same as before
            else: raise ValueError("Please use an accepted distribution type")
        return VAR*np.sqrt(h)
    else:
        r.rename("r", inplace=True) # the input must be a pandas series
        data = pd.DataFrame(r)
        data["mean"] = data["r"].rolling(win, min_periods=start).mean()*h
        data["std"] = data["r"].rolling(win, min_periods=start).std()*np.sqrt(h)
        data["quant"] = data["r"].rolling(win, min_periods=start).quantile(alpha)
        # use .shift(1) to use data up to t-1 (today) to compute VaR associated to date t (tomorrow)
        if garch == False:
            if dist == "gaussian": data["VaR"] = -position*(data["mean"].shift(1)+sp.norm.ppf(alpha)*data["std"].shift(1))
            elif dist == "tstudent": data["VaR"] = -position*(data["mean"].shift(1)+sp.t.ppf(alpha, df)*data["std"].shift(1))
            elif dist == "hist": data["VaR"] = (-position*data["quant"].shift(1))*np.sqrt(h) 
            else: raise ValueError("Please use an accepted distribution type")
        else: 
            data["garch_vol"] = garch11_vol_fixedspec(r, start) 
            if dist == "gaussian": data["VaR"] = -position*(data["mean"].shift(1)+sp.norm.ppf(alpha)*data["garch_vol"].shift(1))
            elif dist == "tstudent": data["VaR"] = -position*(data["mean"].shift(1)+sp.t.ppf(alpha, df)*data["garch_vol"].shift(1))
            elif dist == "hist": data["VaR"] = (-position*data["quant"].shift(1))*np.sqrt(h)
            else: raise ValueError("Please use an accepted distribution type")
        return data["VaR"]  # there are NaN values at the beginning based on start

def ES_basic(r, alpha=0.01, h=1, dist="gaussian", position=1.0, df=5, rolling=False, start=60, win=252, garch=False):
    """
    Compute Expected Shortfall.
    """
    if rolling == False: 
        mean = (r.mean())*h
        std = (r.std())*np.sqrt(h)
        quant = np.quantile(r, alpha)   # for quantile pass directly h-days returns
        if garch == False:
            if dist == "gaussian": ES = alpha**-1 *sp.norm.pdf(sp.norm.ppf(alpha))*std - mean
            elif dist == "tstudent": ES = -1/alpha * (1-df)**(-1) * (df - 2 + sp.t.ppf(alpha, df)**2) * sp.t.pdf(sp.t.ppf(alpha, df), df)*std - mean
            elif dist == "hist": ES = (-(r[r<quant].mean()))*np.sqrt(h)  # minus in front to get positive ES value
            else: raise ValueError("Please use an accepted distribution type")
        else:
            vol = garch11_vol_fixedspec(r, start)
            vol = float(vol.iloc[-1])*np.sqrt(h)
            if dist == "gaussian": ES = alpha**-1* sp.norm.pdf(sp.norm.ppf(alpha))*vol - mean
            elif dist == "tstudent": ES = -1/alpha * (1-df)**(-1) * (df - 2 + sp.t.ppf(alpha, df)**2) * sp.t.pdf(sp.t.ppf(alpha, df), df)*vol - mean
            elif dist == "hist": ES = (-(r[r<quant].mean()))*np.sqrt(h)
            else: raise ValueError("Please use an accepted distribution type")
        return ES*position
    else:
        r.rename("r", inplace=True) # the input must be a pandas series
        data = pd.DataFrame(r)
        data["mean"] = (data["r"].rolling(win, min_periods=start).mean())*h
        data["std"] = (data["r"].rolling(win, min_periods=start).std())*np.sqrt(h)
        data["quant"] = data["r"].rolling(win, min_periods=start).quantile(alpha)
        # use .shift(1) to use data up to t-1 (today) to compute VaR associated to date t (tomorrow)
        if garch == False:
            if dist == "gaussian": data["ES"] = alpha**-1 * sp.norm.pdf(sp.norm.ppf(alpha))*data["std"].shift(1) - data["mean"].shift(1)
            elif dist == "tstudent": data["ES"] = -1/alpha* (1-df)**(-1) * (df - 2 + sp.t.ppf(alpha, df)**2) * sp.t.pdf(sp.t.ppf(alpha, df), df) * data["std"].shift(1) - data["mean"].shift(1)
            elif dist == "hist": 
                es = np.zeros(len(r))
                for i in range(start+1, len(r)):
                    if i<win:
                        ser = r.iloc[:i-1]
                        quant = data["quant"].iloc[i-1]
                        es[i] = ser[ser < quant].mean()
                    else:
                        ser = r.iloc[(i-win):(i-1)]
                        quant = data["quant"].iloc[i-1]
                        es[i] = ser[ser < quant].mean()
                es[es==0] = np.nan
                data["ES"] = (-es)*np.sqrt(h)
            else: raise ValueError("Please use an accepted distribution type")
        else: 
            data["garch_vol"] = garch11_vol_fixedspec(r, start)*np.sqrt(h)
            if dist == "gaussian": data["ES"] = alpha**-1 * sp.norm.pdf(sp.norm.ppf(alpha))*data["garch_vol"].shift(1) - data["mean"].shift(1)
            elif dist == "tstudent": data["ES"] = -1/alpha*(1-df)**(-1) * (df - 2 + sp.t.ppf(alpha, df)**2) * sp.t.pdf(sp.t.ppf(alpha, df), df) * data["garch_vol"].shift(1) - data["mean"].shift(1)
            elif dist == "hist": 
                es = np.zeros(len(r))
                for i in range(start+1, len(r)):
                    if i<win:
                        ser = r.iloc[:i-1]
                        quant = data["quant"].iloc[i-1]
                        es[i] = ser[ser < quant].mean()
                    else:
                        ser = r.iloc[(i-win):(i-1)]
                        quant = data["quant"].iloc[i-1]
                        es[i] = ser[ser < quant].mean()
                es[es==0] = np.nan
                data["ES"] = (-es)*np.sqrt(h)
            else: raise ValueError("Please use an accepted distribution type")
        return data["ES"]*position  # there are NaN values at the beginning based on start

def garch11_vol_fixedspec(r, start): 
    """
    Define the optimal GARCH(1,1) specification given return data and fit the model. 
    - r = pandas series of returns used to fit the garch(1,1) model
    - start = integer number of observations from which 1-step-ahead forecasts are produced
    OUTPUT: dataframe with forecasted volatility, 10% and 5% signals from training data.
    """
    # Set mean and underlying distribution of the GARCH model
    mean_garch = None
    dist_garch = "Normal"
    # Test mean
    pval = ttest_1samp(r, popmean=0)[1]  # if p-value > 0.05 the mean is != 0
    # Test Autocorrelation
    dw = durbin_watson(r)
    if pval < 0.05 and dw > 1.7 and dw < 2.3: mean_garch = "AR"
    elif pval < 0.05 and dw < 1.7 or dw > 2.3: mean_garch = "Constant"
    elif pval > 0.05: mean_garch = "Zero"
    # Test normality
    norm = jarque_bera(r)[1]
    if norm < 0.05: dist_garch = "studentst"
    # Fit the Model (require to scale *100)
    garchmodel = ac.univariate.arch_model(y=r*100, mean=mean_garch, vol="GARCH", p=1, o=0, q=1, dist=dist_garch) 
    modelfit = garchmodel.fit(disp="off", last_obs=r.index[start])
    # Forecast volatility for tomorrow
    forecast = modelfit.forecast(horizon=1)
    v_forecast = forecast.variance
    std_forecast = np.sqrt(v_forecast)/100    # Volatility forecast without *100 scale
    return std_forecast

def pf_gaussianVaRES(r, w, alpha=0.01, h=1, dist="gaussian", position=1.0, df=5, rolling=False, start=60, win=252):
    """
    Compute the portfolio VaR based on gaussian distribution, given the portfolio components and weights.
    - r = pandas dataframe with returns of the assets that compose the portfolio.
    - w = list of weights of the portfolio
    """
    w = np.array(w)
    if rolling==False:
        mean = h*(r.mean())
        cov = np.sqrt(h)*(r.cov())
        vol_pf = np.dot(w.T, np.dot(cov, w))**(1/2)
        if dist == "gaussian": 
            VAR = -position* (np.dot(w.T, mean) + sp.norm.ppf(alpha)*(vol_pf))
            ES = position* (alpha**-1 * sp.norm.pdf(sp.norm.ppf(alpha))*vol_pf - mean)
        elif dist == "tstudent": 
            VAR = -position* (np.dot(w.T, mean) + sp.t.ppf(alpha, df)*(vol_pf))
            ES = position* (-1/alpha * (1-df)**(-1) * (df - 2 + sp.t.ppf(alpha, df)**2) * sp.t.pdf(sp.t.ppf(alpha, df), df)*vol_pf - mean)
        else: raise ValueError("Please use an accepted distribution type")
        return VAR, ES
    else:
        var = np.zeros(len(r))
        es = np.zeros(len(r))
        for i in range(start+1, len(r)):
            if i<win:
                ser = r.iloc[:i-1, :]
                mean = h*(ser.mean())
                cov = np.sqrt(h)*(ser.cov())
                vol_pf = (np.dot(w.T, np.dot(cov, w))**(1/2))*np.sqrt(h)
                if dist == "gaussian": 
                    var[i] = -position*(np.dot(w.T, mean) + sp.norm.ppf(alpha)*(vol_pf))
                    es[i] = position* (alpha**-1 * sp.norm.pdf(sp.norm.ppf(alpha))*vol_pf - mean)
                elif dist == "tstudent": 
                    var[i] = -position*(np.dot(w.T, mean) + sp.t.ppf(alpha, df)*(vol_pf))
                    es[i] = position* (-1/alpha * (1-df)**(-1) * (df - 2 + sp.t.ppf(alpha, df)**2) * sp.t.pdf(sp.t.ppf(alpha, df), df)*vol_pf - mean)
                else: raise ValueError("Please use an accepted distribution type")
            else:
                ser = r.iloc[(i-win):(i-1), :]
                mean = h*(ser.mean())
                cov = np.sqrt(h)*(ser.cov())
                vol_pf = (np.dot(w.T, np.dot(cov, w))**(1/2))*np.sqrt(h)
                if dist == "gaussian": 
                    var[i] = -position* (np.dot(w.T, mean) + sp.norm.ppf(alpha)*(vol_pf))
                    es[i] = position* (alpha**-1 * sp.norm.pdf(sp.norm.ppf(alpha))*vol_pf - mean)
                elif dist == "tstudent": 
                    var[i] = -position* (np.dot(w.T, mean) + sp.t.ppf(alpha, df)*(vol_pf))
                    es[i] = position* (-1/alpha * (1-df)**(-1) * (df - 2 + sp.t.ppf(alpha, df)**2) * sp.t.pdf(sp.t.ppf(alpha, df), df)*vol_pf - mean)
                else: raise ValueError("Please use an accepted distribution type")
        var[var==0] = np.nan
        es[es==0] = np.nan
        r["VaR"] = var
        r["ES"] = es
        return r["VaR"], r["ES"]

def pf_factVaRES(r, fact, w, alpha=0.01, h=1, dist="gaussian", position=1.0, df=5, rolling=False, start=60, win=252):
    """
    VaR is computed using a beta coefficient for each assets included in the portfolio.
    - r = pandas dataframe with returns of the assets of the portfolio.
    - fact = pandas dataframe with returns of the factors portfolios.
    - w = weigths of each asset (or sub-portfolio) in the entire portfolio.
    In r you can pass all the specific assets or fewer sub-portfolios composed by similar-highly correlated assets 
    """
    w = np.array(w)
    if rolling == False:
        mean = r.mean()*h
        mean_pf = np.dot(w.T, mean)
        cov_fact = fact.cov()*np.sqrt(h)
        beta = np.zeros((len(r.columns), len(fact.columns)))  # prepare the matrix
        for i in range(len(r.columns)):
            y = r.iloc[:, i]
            X = fact
            X = sm.add_constant(X)
            reg = sm.OLS(y, X)
            results = reg.fit()
            #print(results.params)
            for j in range(len(fact.columns)):
                beta[i, j] = results.params[j+1]
            #print(beta)
        vol_pf = (np.dot(w.T, np.dot(beta, np.dot(cov_fact, np.dot(beta.T, w)))))**(1/2)
        if dist == "gaussian": 
            VAR = -position*(mean_pf + sp.norm.ppf(alpha)*vol_pf)
            ES = position* (alpha**-1 * sp.norm.pdf(sp.norm.ppf(alpha))*vol_pf - mean_pf)
        elif dist == "tstudent": 
            VAR = -position*(mean_pf + sp.t.ppf(alpha, df)*vol_pf)
            ES = position* (-1/alpha * (1-df)**(-1) * (df - 2 + sp.t.ppf(alpha, df)**2) * sp.t.pdf(sp.t.ppf(alpha, df), df)*vol_pf - mean_pf)
        else: raise ValueError("Please use an accepted distribution type")
        return VAR, ES
    else:
        VAR = np.zeros(len(r))
        ES = np.zeros(len(r))
        for k in range(start+1, len(r)):
            if k<win:
                r_ = r.iloc[:k-1, :]
                fact_ = fact.iloc[:k-1, :]
                mean = r_.mean()*h
                mean_pf = np.dot(w.T, mean)
                cov_fact = fact_.cov()*np.sqrt(h)
                beta = np.zeros((len(r.columns), len(fact.columns)))
                for i in range(len(r.columns)):
                    y = r_.iloc[:, i]
                    X = fact_
                    X = sm.add_constant(X)
                    reg = sm.OLS(y, X)
                    results = reg.fit()
                    for j in range(len(fact.columns)):
                        beta[i, j] = results.params[j+1]
                vol_pf = (np.dot(w.T, np.dot(beta, np.dot(cov_fact, np.dot(beta.T, w)))))**(1/2)
                if dist == "gaussian": 
                    VAR[k] = -position*(mean_pf + sp.norm.ppf(alpha)*vol_pf)
                    ES[k] = position* (alpha**-1 * sp.norm.pdf(sp.norm.ppf(alpha))*vol_pf - mean_pf)
                elif dist == "tstudent": 
                    VAR[k] = -position*(mean_pf + sp.t.ppf(alpha, df)*vol_pf)
                    ES[k] = position* (-1/alpha * (1-df)**(-1) * (df - 2 + sp.t.ppf(alpha, df)**2) * sp.t.pdf(sp.t.ppf(alpha, df), df)*vol_pf - mean_pf)
                else: raise ValueError("Please use an accepted distribution type")
            else:
                r_ = r.iloc[(k-win):(k-1), :]
                fact_ = fact.iloc[(k-win):(k-1), :]
                mean = r_.mean()*h
                mean_pf = np.dot(w.T, mean)
                cov_fact = fact_.cov()*np.sqrt(h)
                beta = np.zeros((len(r.columns), len(fact.columns)))
                for i in range(len(r.columns)):
                    y = r_.iloc[:, i]
                    X = fact_
                    X = sm.add_constant(X)
                    reg = sm.OLS(y, X)
                    results = reg.fit()
                    for j in range(len(fact.columns)):
                        beta[i, j] = results.params[j+1]
                vol_pf = (np.dot(w.T, np.dot(beta, np.dot(cov_fact, np.dot(beta.T, w)))))**(1/2)
                if dist == "gaussian": 
                    VAR[k] = -position*(mean_pf + sp.norm.ppf(alpha)*vol_pf)
                    ES[k] = position* (alpha**-1 * sp.norm.pdf(sp.norm.ppf(alpha))*vol_pf - mean_pf)
                elif dist == "tstudent": 
                    VAR[k] = -position*(mean_pf + sp.t.ppf(alpha, df)*vol_pf)
                    ES[k] = position* (-1/alpha * (1-df)**(-1) * (df - 2 + sp.t.ppf(alpha, df)**2) * sp.t.pdf(sp.t.ppf(alpha, df), df)*vol_pf - mean_pf)
                else: raise ValueError("Please use an accepted distribution type")
        VAR[VAR==0] = np.nan
        ES[ES==0] = np.nan
        VAR = pd.Series(VAR, index=r.index, name="VaR")
        ES = pd.Series(ES, index=r.index, name="ES")
        r = r.merge(VAR, right_index=True, left_index=True)
        r = r.merge(ES, right_index=True, left_index=True)
        return r["VaR"], r["ES"]

def VaR_validation(r, VaR, alpha=0.01, alpha_test=0.05, print_output=False):
    """
    - r = pandas series of returns
    - VaR = pandas series with VaR values (!!must be positive!!)
    - alpha_test = significance level for validation test
    """
    T = len(r)
    # VaR with - becasue we need it has a loss here
    df = pd.merge(r, -VaR, right_index=True, left_index=True)
    theo_exc = T*alpha
    # lenght of the df that contains only datapoints where returns are below VaR level
    real_exc = len(df.iloc[:, 1][df.iloc[:, 0]<df.iloc[:, 1]])
    zscore = (real_exc-theo_exc)/(np.sqrt(alpha*(1-alpha)*T))
    pval = sp.norm.cdf(-abs(zscore))
    pval = pval*2
    if print_output == True:
        print("-----------------------------------")
        print(f"Theoretical exceedances: {theo_exc}")
        print(f"Realized exceedances: {real_exc}")
        print("--> Zscore = ", zscore.round(4))
        print("--> pvalue = ", pval.round(4))
        if pval>alpha_test: print("The VaR is valid")
        else: print("The VaR is NOT valid")
        print("-----------------------------------")
    return zscore, pval

def VEV(VaR, T):
    """
    - VaR = unique float VaR value
    - T = number of trading days from which the VaR is calculated
    Assume alpha = 0.025 for the VaR 
    """
    VEV = (np.sqrt(3.842-2*np.log(VaR))-1.96)/np.sqrt(T)
    return VEV
