# Rolling Drawdow
import numpy as np
import pandas as pd

def drawdown(ret_data, window=6):
    import numpy as np
    import pandas as pd
    tickers = ret_data.columns
    dd = pd.DataFrame(index=ret_data.index)
    for tick in tickers:
        roll_sum = ret_data[tick].rolling(window, min_periods=1).sum()
        roll_mdd = roll_sum.rolling(1).min()
        roll_mdd[roll_mdd > 0.0] = 0.0
        dd[tick] = roll_mdd
    return dd

# Information ratio
def info_ratio(ret_data, avg_ret_data):
    import numpy as np
    import pandas as pd
    index_tick = ret_data.columns[0]  #Index fund must be in position zero
    qq = pd.DataFrame(index=ret_data.index)
    IR = []
    for tick in ret_data.columns:
        qq["diff_sp_{}".format(tick)] = (ret_data[tick] - ret_data[index_tick])
        IR.append((avg_ret_data[tick] - avg_ret_data[index_tick])/(qq["diff_sp_{}".format(tick)].std()))
    return IR

def comp_ann_gr_rate(data_price):
    import numpy as np
    import pandas as pd
    ann = data_price.resample("A").last()
    ann_ret = np.log(1 + ann.pct_change())
    ann_ret = ann_ret.dropna()
    return (ann_ret.sum()/len(ann_ret))

def comp_ann_gr_rate_noprice(data_ret):
    import numpy as np
    import pandas as pd
    average_ret = np.mean(data_ret)
    return (average_ret*12)

def historical_VaR_CVaR(data_ret, cl=99):
    import numpy as np
    import pandas as pd
    VaR_list = []
    CVaR_list = []
    for tick in data_ret.columns:
        VaR = np.percentile(data_ret[tick], (100-cl))
        VaR_list.append(VaR)
        CVaR_list.append(np.mean(data_ret[tick][data_ret[tick] < VaR]))
    return [-i for i in VaR_list], [-i for i in CVaR_list]

def traditional_metrics(data_ret, data_price, no_price=False):
    import numpy as np
    import pandas as pd
    metrics = pd.DataFrame(index=data_ret.columns)
    if no_price == False:
        cagr = comp_ann_gr_rate(data_price)
    else:
        cagr = comp_ann_gr_rate_noprice(data_ret)
    # Monthly Average return
    avg = data_ret.mean()
    # Monthly Volatility
    stdev = data_ret.std()
    # Sharpe 
    SH = avg/stdev
    # IR
    INFO_RATIO = info_ratio(data_ret, avg)
    # Sortino
    neg_stdev = (data_ret[data_ret < 0.0]).std()
    SR = avg/neg_stdev
    # VArCvar
    v, cv = historical_VaR_CVaR(data_ret, cl=95)
    # Drawdown
    dd = drawdown(data_ret)
    min_dd = dd.min()

    metrics["CAGR"] = cagr.values
    metrics["Monthly Return"] = avg.values
    metrics["Monthly Volatility"] = stdev.values
    metrics["Sharpe"] = SH.values
    metrics["Sortino"] = SR.values
    metrics["IR"] = INFO_RATIO
    metrics["VaR"] = v
    metrics["CVaR"] = cv
    metrics["Max Drawdown"] = min_dd.values
    return metrics

def avg_performance_drag(df_sp_strat, inputs, sp_tick):
    """
    Return the difference in average return between the strategy and the S&P 500 index
    (positie value means outperformance)
    """
    import numpy as np
    import pandas as pd
    pf_dr = []
    pf_dr.append(0)
    for i in inputs: 
        drag = np.mean(df_sp_strat[i]) - np.mean(df_sp_strat[sp_tick])
        #print("Performance drag for {}: ".format(i), drag.round(4)*100, "%")
        pf_dr.append(drag)
    return pf_dr

def bull_drag(df_sp_strat, inputs, sp_tick, threshold):
    """
    Return the difference in total return between the strategy and the S&P 500 index
    only when the S&P returns are above the threshold value 
    (positie value means outperformance)
    """
    import numpy as np
    import pandas as pd
    bu_dr = []
    bu_dr.append(0)
    for inp in inputs: 
        use_sp = []
        use_st = []
        for i in range(len(df_sp_strat)):
            if (df_sp_strat[sp_tick].iloc[i] > threshold) == True:
                use_sp.append(df_sp_strat[sp_tick].iloc[i])
                use_st.append(df_sp_strat[inp].iloc[i])
        use_sp = np.array(use_sp)
        use_st = np.array(use_st)
        dr = use_st - use_sp
        drag = np.mean(dr)
        bu_dr.append(drag)
        #print("Bull drag for {}: ".format(inp), drag.round(4)*100, "%")
    return bu_dr

def certainty(df_sp_strat, inputs, sp_tick, threshold):
    """
    Return the percentage of time the strategy returns are positive
    given S&P 500 returns below the loss threshold
    """
    import numpy as np
    import pandas as pd
    cr = []
    cr.append(0)
    #print("{}% S&P 500 loss threshold".format(threshold))
    #print("--------------------------")
    for inp in inputs: 
        sig = []
        for i in range(len(df_sp_strat)):
            if (df_sp_strat[sp_tick].iloc[i] <= threshold) == True:
                if (df_sp_strat[inp].iloc[i] > 0.0) == True:
                    sig.append(1)
                else:
                    sig.append(0)
        sig = np.array(sig)
        pct_certainty = (sig.sum())/len(sig)
        cr.append(pct_certainty)
        #print("Certainty for {}: ".format(inp), pct_certainty.round(4)*100, "%")
    return cr 

def avg_tail_return(df_sp_strat, threshold):
    """
    Return the average return of the strategy
    given S&P 500 returns below the loss threshold
    """
    import numpy as np
    import pandas as pd
    av = []
    #print("{}% S&P 500 loss threshold".format(threshold))
    #print("--------------------------")
    sig = []
    for inp in df_sp_strat.columns: 
        for i in range(len(df_sp_strat)):
            if (df_sp_strat.iloc[i, 0] <= threshold) == True:
                #Average only in the case where returns of the strategy are positive
                #if (df_sp_strat[inp].iloc[i] > 0.0) == True:
                    #sig.append(df_sp_strat[inp].iloc[i]) #comment out the next line if you use this one
                sig.append(df_sp_strat[inp].iloc[i])
        avg = np.mean(sig)
        av.append(avg)
        #print("Average tail loss for {}: ".format(inp), avg.round(4)*100, "%")
    return av

def specific_metrics(data_ret, th_bull, th_cr, th_avg_tail):
    import numpy as np
    import pandas as pd
    metrics = pd.DataFrame(index=data_ret.columns)
    inputs = data_ret.columns[1:]
    sp_tick = data_ret.columns[0]   #target index must be in position 0
    p = avg_performance_drag(data_ret, inputs=inputs, sp_tick=sp_tick)
    b = bull_drag(data_ret, inputs=inputs, sp_tick=sp_tick, threshold=th_bull)
    c = certainty(data_ret, inputs=inputs, sp_tick=sp_tick, threshold=th_cr)
    a = avg_tail_return(data_ret, threshold=th_avg_tail)
    metrics["Avg Performance Drag"] = p
    metrics["Bull Drag"] = b
    metrics["Certainty"] = [i*100 for i in c]
    metrics["Avg Tail return"] = a
    return metrics
