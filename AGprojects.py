import streamlit as st
import pandas as pd
import pandas_datareader as web
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from FunctionsInvCha_web import *
from ClassOptions import *
from FunctionsThesis import *
from FunctionsHRP import *

import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import plotly.express as px


########### Home and Sidebar ###################################################
st.sidebar.header("AG Projects")
st.sidebar.subheader("LinkedIn:")
st.sidebar.write("www.linkedin.com/in/alessiogiust67")
st.sidebar.subheader("GitHub:")
st.sidebar.write("https://github.com/AleG67")
st.sidebar.subheader("Navigate trough my Projects")
select_projects = st.sidebar.selectbox("Choose the project you want to see:", 
                                        options=("Home", "Investment challenge", "Thesis report", "HRP Portfolio", "Options Valuation"))
if select_projects == "Home":
    st.title("Welcome to AG Projects")
    st.subheader("About me")
    st.markdown("I'm an italian MS student graduating in Finance (Double Degree program between the University of Padova and Zicklin School of Business (CUNY)), holding a Bachelor's Degree in Economics. I'm passionate about Quantitative Finance, Risk management and Asset management. I'm also very interested in coding and Data Science applied to Finance. Here you can find some simple projects that I developed to improve my coding skills (I learned to code on my own so please understand that there might be errors or bad-written code).")
    st.subheader("Contacts")
    st.markdown("LinkedIn: \n www.linkedin.com/in/alessiogiust67")
    st.markdown("GitHub: \n https://github.com/AleG67")
    st.markdown("GitHub for Streamlit code: \n https://github.com/AleG67/StreamlitApp_AGprojects")
@st.cache
def import_data(filename):
    data = pd.read_excel(filename)
    return data
    
########### Project 1: Investment challenge ############################
if select_projects == "Investment challenge":
        # CACHE = avoid that actions run more than once every time the user inputs something new in the page
    @st.cache
    def get_stock(ticker, start, end):
        data = web.DataReader(ticker, "yahoo", start=start, end=end) 
        data["MA5"] = data["Adj Close"].rolling(5).mean()
        data["MA20"] = data["Adj Close"].rolling(20).mean()
        data = data.dropna()
        return data

    # Create containers
    header = st.beta_container()
    esg_stocks = st.beta_container()
    #stock_chart = st.beta_container()
    indicators_models = st.beta_container()
    prediction = st.beta_container()
    
    # Add things into the containers
    with header:
        #add an image
        st.image("https://i.ytimg.com/vi/7UsyG51Eog8/maxresdefault.jpg")
        #st.title("Reply Sustainable Investment Challenge")     
        st.title("Predict daily stock movements with ML models")
        st.write("**GitHub repository link:** https://github.com/AleG67/ReplyInvestmentChallenge")
        st.markdown("In this project I'm predicting stock price movements with four machine learning classification methods. This is a very basic coding exercise to experience the practical aspects and problems of implementing ML techniques on real time data.")
        st.markdown("In the project there are a number of semplifications, such as: \n - Basic machine learning models. \n - Basic data sources and input features. \n - Semplifications in the trading process and no automated trading.")
        st.markdown("This is simply intended to show a basic framework for a quantitative trading strategy, the models are not sophisticated enough to be effective and the strategy management is not realistic.")

    with esg_stocks:
        st.header("I. Tickers used in the models and ESG rating")
        st.markdown("The rule of the challenge was to assign a bonus if you were long on companies with good ESG rating and a penalty otherwise. Hence an idea could be to screen the list of stocks provided in the challenge based on the accuracy scores of each classification method. Then trade based on the predictions and their ESG rating, based on 3 categories: \n  - Good (AAA, AA) \n  - Average (A, BBB, BB, B) \n  - Bad (B, CCC)")
        # import a dataset and show it in the page
        esg_stock_data = import_data("dataneeded/final_esg.xlsx")
        st.write(esg_stock_data)

    with indicators_models:
        st.header("II. Technical Indicator inputs and Models deployed")
        # Create 2 columns (then to add things in them use col1. or col2. instead of st.)
        col1, col2 = st.beta_columns(2)
        col1.subheader("Features utilized")
        # create bullet points with the features
        col1.markdown("* **S&P 500 Up/Down: ** Direction of the S&P returns for today.")
        col1.markdown("* **S&P 500 Volatility: ** Rolling 10-days volatility of the S&P.")
        col1.markdown("* **RSI: ** Relative strenght index.")
        col1.markdown("* **SR: ** Stochastic oscillator.")
        col1.markdown("* **MACD: ** Moving average convergence-divergence.")
        col1.markdown("* **ATR: ** Average true range.")
        #add more
        col2.subheader("Machine Learning models")
        col2.markdown("* **Random Forest**")
        col2.markdown("* **XG Boost**")
        col2.markdown("* **Linear Discriminant Analysis (LDA)**")
        col2.markdown("* **Quadratic Discriminant Analysis (QDA)**")

    with prediction:
        st.header("III. Models implementation on daily market data")
        st.subheader("An error will occur if the market is closed when you try to obtain the prediction, because the models require current market data to predict.")
        # select box
        box = st.selectbox("Choose the stock you want to obtain predictions for:", options=esg_stock_data["Ticker"])
        st.subheader(f"Models Predictions for {box}:")
        rf, xg, lda, qda = model_application_for_web(box)
        st.markdown(f"* **Random Forest prediction:** {rf}")
        st.markdown(f"* **XGBoost prediction:** {xg}")
        st.markdown(f"* **LDA prediction:** {lda}")
        st.markdown(f"* **QDA prediction:** {qda}")

########### Project 2: Thesis Report #################################
# import data Direct and Indirect
@st.cache
def get_data_thesis():
    strategies = pd.read_excel("Dataneeded/Data_tesi.xlsx", sheet_name="Monthly against tail risk", index_col="Dates")
    strategies_daily = pd.read_excel("Dataneeded/Data_tesi.xlsx", sheet_name="Daily against tail risk", index_col="Dates")
    strat = strategies[["SPXT", "PPUT", "CLL", "CLLZ", "VXTH", "SPVQDTR", "LOVOL"]].dropna()
    strat_ret = np.log(1+strat.pct_change()).dropna()
    strat_daily = strategies_daily[["SPXT", "PPUT", "CLL", "CLLZ", "VXTH", "SPVQDTR", "LOVOL"]].dropna()
    strat_daily_ret = np.log(1+strat_daily.pct_change()).dropna()
    controlvol = strategies[["SPXT", "SP5LVIT", "SPXT15UT", "SPLV15UT"]].dropna()
    controlvol_ret = np.log(1+controlvol.pct_change()).dropna()
    controlvol_ret = controlvol_ret[86:].copy()
    # import data other, my Veq and my Garch
    asset_ret = pd.read_excel("Dataneeded/Other.xlsx", index_col="Dates")
    veq = pd.read_excel("Dataneeded/MyVEQ.xlsx", index_col="Dates")
    garch = pd.read_excel("Dataneeded/MyGARCH.xlsx", index_col="Dates")
    # Import performance metrics
    trad = pd.read_excel("Dataneeded/SummTrad.xlsx", index_col="Unnamed: 0")
    spec = pd.read_excel("Dataneeded/SummSpec.xlsx", index_col="Unnamed: 0")
    T = pd.read_excel("Dataneeded/GstatT.xlsx", index_col="Unnamed: 0")
    S = pd.read_excel("Dataneeded/GstatS.xlsx", index_col="Unnamed: 0")
    return strat_ret, strat_daily_ret, controlvol_ret, asset_ret, veq, garch, trad, spec, T, S
strat_ret, strat_daily_ret, controlvol_ret, asset_ret, veq, garch, trad, spec, T, S = get_data_thesis()

if select_projects == "Thesis report":
    sns.set()
    # Create containers
    header = st.beta_container()
    direct = st.beta_container()
    indirect = st.beta_container()
    other = st.beta_container()
    performance = st.beta_container()
    myindexVEQTOR = st.beta_container()
    myindexGARCH = st.beta_container()

    with header:
        st.title("Tail Risk Protection strategies")
        st.write("**GitHub repository link:** https://github.com/AleG67/ThesisCode")
        st.markdown("In my Master's Thesis I analyzed different strategies to protect portfolios and market index investments against extreme losses. Here you can find a quick summary of the existing strategies I analyzed and the one I created. If interested you can find my Thesis here: .")
        st.markdown("In the text input section below you can choose the start and end date to compare the different strategies over time in different sub-periods. The given inputs are the start and end for the widest time period available to compare all the indexes based on data availability.")
        start = st.text_input("Starting date in format YYYY-MM-DD", "2006-04-28")
        end = st.text_input("Ending date in format YYYY-MM-DD", "2020-12-31")
    
    with direct:
        st.title("Direct Tail Risk Protection indexes")
        st.markdown(" - **PPUT:** SP500 and 5% OTM Put Options. \n - **CLL:** Collar strategy, 5% OTM Puts costs partially compensated by selling 10% OTM calls. \n - **CLLZ:** Collar strategy where 5% OTM Puts bought are entirely financed by selling 10% OTM Calls. \n - **VXTH:** SP500 and OTM 30-delta call options on the VIX index (weights based on the VIX futures levels). \n - **SPVQDTR:** later called VEQTOR, index that allocates to SP500, short term VIX futures or cash, with weights based on two monthly volatility signals (realized volatility and implied volatility trend) plus a stop loss feature that shifts the allocation entirely to cash if the index is in a -2%, 5-days losing streak. (The tail protection strategy that I’ll apply to other equity indexes is based on the signals used to create this index, with small changes). \n - **LOVOL:** combination of VXTH and BXM (CBOE “BuyWrite” index, S&P 500 long position plus selling covered OTM calls).")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(strat_ret["SPXT"].loc[start:end].cumsum(), "black", linewidth=2)
        ax.plot(strat_ret["PPUT"].loc[start:end].cumsum(), "purple")
        ax.plot(strat_ret["CLL"].loc[start:end].cumsum(), "m")
        ax.plot(strat_ret["CLLZ"].loc[start:end].cumsum(), "pink")
        ax.plot(strat_ret["VXTH"].loc[start:end].cumsum(), "blue")
        ax.plot(strat_ret["SPVQDTR"].loc[start:end].cumsum(), "green")
        ax.plot(strat_ret["LOVOL"].loc[start:end].cumsum(), "c")
        ax.legend(strat_ret.columns)
        ax.set_title("Comparison of direct tail protection indexes", fontsize=20, fontweight="bold")
        st.pyplot(fig)

    with indirect:
        st.title("Indirect Tail Risk Protection indexes (Managed volatility)")
        st.markdown(" - **SP5LVIT:** S&P 500 Low Volatility index. It measures the performance of the 100 least volatile stocks in the S&P 500. \n - **SPXT15UT:** S&P 500 total return Index with a risk control target of 15% (later called SP 15). \n - **SPLV15UT:** S&P 500 Low Volatility index with Daily Risk Control target of 15% (later called SP 15 LowVol).")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(controlvol_ret["SPXT"].loc[start:end].cumsum(), "black", linewidth=2) 
        ax.plot(controlvol_ret["SP5LVIT"].loc[start:end].cumsum(), "darkorange")
        ax.plot(controlvol_ret["SPXT15UT"].loc[start:end].cumsum(), "darkgoldenrod")
        ax.plot(controlvol_ret["SPLV15UT"].loc[start:end].cumsum(), "red")
        ax.legend(controlvol_ret.columns)
        ax.set_title("Comparison of volatility-controlled indexes", fontsize=20, fontweight="bold")
        st.pyplot(fig)

    with other:
        st.title("Other assets for Tail Risk Protection")
        st.markdown(" - **BARCCTA:** managed futures index. \n - **DBVELVIS:** long volatility investment strategy index. \n - **SPVXSTR:** index that tracks 1-month VIX futures. \n - **SPVXMTR:** index that tracks VIX futures with 5-months average maturity.")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(asset_ret["SPXT"].loc[start:end].cumsum(), "black",label="SPXT")
        ax.plot(asset_ret["BARCCTA-SPXT"].loc[start:end].cumsum(), label="Index with BARCCTA")
        ax.plot(asset_ret["DBVELVIS-SPXT"].loc[start:end].cumsum(), label="Index with DBVELVIS")
        ax.plot(asset_ret["SPVXSTR-SPXT"].loc[start:end].cumsum(), label="Index with SPVXSTR")
        ax.plot(asset_ret["SPVXMTR-SPXT"].loc[start:end].cumsum(), label="Index with SPVXMTR")
        ax.legend()
        ax.set_title("Comparison of indexes with alternative assets", fontsize=20, fontweight="bold")
        st.pyplot(fig)

    with performance:
        st.title("Performance Metrics")
        st.markdown(" - **Performance drag:** difference between the average monthly return of the index that include tail risk protection and the S&P. \n - **Bull drag:** average difference between the monthly returns of the index that include tail risk protection and the S&P, only in months where the S&P posts positive returns. \n - **Certainty:** percentage of times the index monthly returns are positive given S&P returns below a given loss threshold. Here 5% S&P 500 loss threshold. \n - **Average Tail return:** average monthly return of the index given S&P returns below a given loss threshold. Here the S&P 500 loss threshold is 10%. This are indexes that have big allocation to SP so small negative return is good.")
        # show dataframes with performance metrics
        st.write(trad)
        st.write(spec)

    with myindexVEQTOR:
        st.title("VEQTOR index with different weights")
        st.markdown("The allocation matrix is based on the paper “Dynamically hedging equity market tail risk using VEQTOR” by Standards & Poor’s, but I increased the allocation to the S&P when the signals indicate a lower level of tail risk and increased the allocation to VIX futures when the implied tail risk is higher.")
        st.markdown("These changes are based on my analysis of all the strategies, that brought to the conclusion that we have to eliminate the performance drag as much as possible when markets are calm, and increase tail protection when we have clear signs of extreme volatility, because volatility clusters so even if we miss out on the first increase we can still benefit from subsequent spikes in volatility. These conclusions are supported by: \n - Large body of literature and practitioners’ consensus on volatility clustering, \n - The fact that to insure against big losses you have to pay premiums that reduce performance, so we try to use market timing to avoid that, \n - My analysis of tail-protection strategies performance in this thesis.")

    with myindexGARCH:
        st.title("Tail Risk protection strategy based on GARCH model")
        st.markdown("The weights are the same as the modified VEQTOR strategy. The signal is created usign the volatility predictions of a GARCH(1,1) model, with a trigger if the volatility forecast exceed a certain threshold. You can more info regarding the strategy in the Thesis.")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(veq["Ret_SPXT"].loc[start:end].cumsum(), "black", label="SPXT")
        ax.plot(veq["Ret_INDEX"].loc[start:end].cumsum(), "darkgreen", label="Modified VEQTOR")
        ax.plot(strat_daily_ret["SPVQDTR"].loc[start:end].cumsum(), "green", linestyle="--", label="Original VEQTOR")
        ax.plot(garch["Ret_INDEX"].loc[start:end].cumsum(), "blue", label="GARCH strat")
        ax.legend()
        ax.set_title("Comparison of VEQTOR and my strategies", fontsize=20, fontweight="bold")
        st.pyplot(fig)
        # Performance metrics for GARCH
        st.write(T.iloc[0:2, :])
        st.write(S.iloc[0:2, :])

########### Project 3: HRP Portfolio #################################
if select_projects == "HRP Portfolio":
    header = st.beta_container()
    stockselection = st.beta_container()
    hcl = st.beta_container()
    qdiagmat = st.beta_container()
    pfw = st.beta_container()

    with header:
        st.title("Hierarchical Risk Parity portfolio")
        st.markdown("Hierarchical Risk Parity is a new portfolio optimization technique proposed by Professor Marcos Lopez De Prado in 'Building diversified portfolios that outperform out-of-sample' (2015), where you can find the detailed description of this technique. Here you can find a basic implementation based on the code provided in the paper.")
        st.markdown("HRP is based on 3 steps: \n - Hierarchical Clustering based on a specific distance matrix. \n - Quasi-diagonalization of the covariance matrix, which reorganizes the rows and columns of the matrix, so that the largest values lie along the diagonal. In this way similar investments are placed together, and dissimilar investments are placed far apart. \n - Recursive bisection, used to compute weights based on inverse-variance allocation on subsets of the quasi-diagonalized matrix and then aggregate them together to get the final weights of the portfolio.")
        st.markdown("The idea behind HRP is to cluster together similar assets and avoid the inversion of the covariance matrix that creates great instability in traditional portfolio optimization techniques (see the paper for more details).")

    with stockselection:
        st.header("Select the assets you want in your portfolio")
        assets = ["AAPL", "TSLA", "MSFT", "AMZN", "JPM", "BAC", "JNJ", "GOOG", "ATVI", "TROW"]
        alternatives = st.multiselect('Which assets do you want? (At least two)', assets,["AAPL", "JPM", "BAC", "JNJ", "GOOG", "MSFT"])
    
    with hcl:
        st.header("1) Hierarchical Clustering")
        # Get stocks data
        stocks = web.DataReader(alternatives, "yahoo", "2018-01-01", "2021-04-01")["Adj Close"]
        # Returns
        ret = stocks.pct_change().dropna()
        cov, corr = ret.cov(), ret.corr()
        dist = correlDist(corr)    # get new measure of distance
        link = sch.linkage(dist, 'single')  # clustering with scipy
        fig = ff.create_dendrogram(dist, labels=alternatives)   # Plot dendrogram with plotly
        fig["layout"].update({"title": "Stocks with similar characteristics are clustered together"})
        st.plotly_chart(fig, use_container_width=True)
    
    with qdiagmat:
        st.header("2) Quasi-Diagonalization")
        sortIx1 = getQuasiDiag(link)            
        sortIx2 = corr.index[sortIx1].tolist() 
        corrQDIAG = corr.loc[sortIx2, sortIx2]
        fig = px.imshow(corrQDIAG)
        fig["layout"].update({"title": "Quasi-Diagonalized corr. matrix (high correlations close to the diagonal)"})
        st.plotly_chart(fig, use_container_width=True)

    with pfw:
        st.header("3) Recursive bisection to obtain the final weights")
        w = getHRP(cov, corr)
        df = pd.DataFrame(w, columns=["HRP weights"])
        col1, col2, col3 = st.beta_columns([1.5,3,1])
        with col1: st.write("")
        with col2: st.write(df)
        with col3: st.write("")

########### Project 4: Options Valuation #################################
if select_projects == "Options Valuation":
    # Create containers
    header = st.beta_container()
    stoch_process = st.beta_container()
    BSMvaluation = st.beta_container()
    MSCvaluation = st.beta_container()

    # me = Market Environment, here you can find the fixed parameters that the user can't change
    me = market_environment('me_gbm', dt.datetime(2020, 1, 1))
    me.add_constant('final_date', dt.datetime(2020, 12, 31)) # 1-year simulation
    me.add_constant('currency', 'EUR')
    me.add_constant('frequency', 'M') # monthly frequency
    me.add_constant('paths', 100)     # 100 paths for MC to be quick
    me.add_constant('maturity', dt.datetime(2020, 12, 31))   # 1-year maturity fixed
    # Fixed parameters for jump diffusion
    me.add_constant('mu', -0.6)
    me.add_constant('delta', 0.1)

    with header:
        st.title("Valuation of Options with BSM formula and Monte Carlo Simulation")
    
    with stoch_process:
        st.header("I. Simulate stochastic processes used to model stock price movements")
        st.markdown("The parameters defined here are used to generate the MC paths and price the options in section II and III.")
        under_price = st.slider("Select the starting price of the underlying asset:", 30.0, 40.0, 35.0, 1.0)
        strike = st.slider("Select the strike price of the option:", 20.0, 60.0, 40.0, 1.0)
        vola = st.slider("Select the monthly volatility of the underlying asset:", 0.05, 0.5, 0.2, 0.05)
        rate = st.slider("Select the risk free rate:", 0.0050, 0.050, 0.020, 0.0050)
        jump_param = st.slider("Select the parameter that regulates the jump intensity (lambda):", 0.1, 0.8, 0.5, 0.1)
        me.add_constant('lambda', jump_param) # significant jumps 
        me.add_constant('initial_value', under_price) # current price of the stock
        me.add_constant('strike', strike)
        me.add_constant('volatility', vola)
        csr = constant_short_rate('csr', rate)    # define the costant short rate (csr)
        me.add_curve('discount_curve', csr)
        # Stochastic processes
        gbm = geometric_brownian_motion('gbm', me) # GBM as the stochastic process to use in the valuation
        jump = jump_diffusion('jump', me)     # Jump Diffusion as the stochastic process to use in the valuation
        gbmpaths = gbm.get_instrument_values()
        jumppaths = jump.get_instrument_values()
        # Plot only 5 paths for each process
        sns.set()
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(gbmpaths[:, :5].round(2), color='blue', label="GBM")
        ax.plot(jumppaths[:, :5].round(2), color='red', label="Jump Diffusion")
        ax.set_title("Comparison of BGM and Jump diffusion monthly paths", fontsize=20, fontweight="bold")
        st.pyplot(fig)

    with BSMvaluation:
        st.header("II. Pricing European Call and Put Options with Black-Scholes-Merton formula")
        st.markdown("Application of the Black-Scholes-Merton formula (without dividends), to price European Call and Put Options.")
        st.write("**Call Option price:** ", bs_eu_option(under_price,strike,1,rate,vola,is_call=True).round(2))
        st.write("Delta of the Call Option: ", delta_option(under_price,strike,1,rate,vola,is_call=True).round(3))
        st.write("**Put Option price:** ", bs_eu_option(under_price,strike,1,rate,vola,is_call=False).round(2))
        st.write("Delta of the Put Option: ", delta_option(under_price,strike,1,rate,vola,is_call=False).round(3))
    
    with MSCvaluation:
        st.header("III. Pricing European Options and other more complex options with Monte Carlo simulation")
        st.markdown("In this case we can price 'plain vanilla' options but also more complex derivatives because using MCS you just need the final payoff of the option to price it. The limitation is that the python code is written in such a way that it is not possible to price instruments that allow early exercise, but still it can be used to price path-dependent options such as....")
        # Select the payoff function (ie the type of option you want to price)
        st.markdown("**Available payoffs:** \n - **Vanilla Call:** np.maximum(maturity_value - strike, 0) \n - **Vanilla Put:** np.maximum(strike - maturity_value, 0), \n - **Simple Asian Call:** np.maximum(mean_value - strike, 0), \n - **Simple Asian Put:** np.maximum(strike - mean_value, 0) \n - **Regular/asian call payoff mix:** np.maximum(0.33*(maturity_value + max_value) - strike, 0), \n - **Lookback with minimum as strike:** np.maximum(maturity_value - min_value, 0)")
        payoff_func = st.selectbox("Select a payoff function:", options=("np.maximum(maturity_value - strike, 0)", 
                                                                        "np.maximum(strike - maturity_value, 0)",
                                                                        "np.maximum(mean_value - strike, 0)",
                                                                        "np.maximum(strike - mean_value, 0)",
                                                                        "np.maximum(0.33*(maturity_value + max_value) - strike, 0)",
                                                                        "np.maximum(maturity_value - min_value, 0)"))
        # Select the stoch process for MC simulation
        process = st.selectbox("Select the underlying stochastic process:", options=("Geometric Brownian Motion", "Jump Diffusion"))
        if process == "Geometric Brownian Motion": under = gbm   
        else: under = jump
        option = valuation_mcs_european('option', underlying=under, mar_env=me, payoff_func=payoff_func)
        st.write("**Present value of the Option:** ", option.present_value().round(2))
        st.write("Delta of the Option: ", option.delta())
