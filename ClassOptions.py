import pandas as pd
import numpy as np
import datetime as dt

#### Code based on the books of Yves Hilpisch (Pyhton for Finance)
############### Black-Scholes-Merton Model Formulas ########################
def bs_eu_option(S, K, t, rf, sigma, is_call = True):
    """
    Price an european option call or put, default is call 
    t = time to maturity, usually (T-t)
    rf = risk free rate
    """
    from scipy import log,exp,sqrt,stats
    d1=(log(S/K)+(rf+sigma*sigma/2.)*t)/(sigma*sqrt(t))
    d2 = d1-sigma*sqrt(t)
    if is_call == True:
        return S*stats.norm.cdf(d1) - K*exp(-rf*t)*stats.norm.cdf(d2)
    else: 
        return K*exp(-rf*t)*stats.norm.cdf(-d2) - S*stats.norm.cdf(-d1)
#print(bs_eu_option(40,40,1,0.03,0.2,is_call=False))

def delta_option(S, K, t, rf, sigma, is_call=True):
    """
    Compute the delta of an Option
    """
    from scipy import log,exp,sqrt,stats 
    d1 = (log(S/K)+(rf+sigma*sigma/2.)*t)/(sigma*sqrt(t))
    if is_call == True:
        return stats.norm.cdf(d1)
    else:
        return stats.norm.cdf(d1)-1
##############################################################################

def get_year_deltas(date_list, day_count=365.):
    """
    Return vector of floats with day deltas in year fractions. Initial value normalized to zero.
    - date_list = collection of datetime objects
    - day_count = number of days for a year (to account for different conventions)
    OUTPUT:
    - delta_list = year fractions
    """
    start = date_list[0]
    delta_list = [(date - start).days / day_count for date in date_list]
    return np.array(delta_list)

############ CLASS TO DEFINE A DISCOUNTING RATE and GET DISCOUNT FACTORS ############
class constant_short_rate(object):
    """
    Class for constant short rate discounting.
    ATTRIBUTES:
    - name = name of the object
    - short_rate = constant rate for discounting
    """
    def __init__(self, name, short_rate): 
        self.name = name
        self.short_rate = short_rate
        if short_rate < 0: 
            raise ValueError("Short rate negative.")  # this is debatable given recent market realities
    
    def get_discount_factors(self, date_list, dtobjects=True): 
        """
        Get discount factors given a list/array of datetime objects or year fractions
        """
        if dtobjects is True:
            dlist = get_year_deltas(date_list)   #Use the function defined above
        else:
            dlist = np.array(date_list)
        dflist = np.exp(self.short_rate * np.sort(-dlist)) 
        return np.array((date_list, dflist)).T
    
############ CLASS to DEFINE A MARKET ENVIRONMENT ############
class market_environment(object):
    """
    Class to model a market environment relevant for valuation.
    ATTRIBUTES:
    - name = name of the market environment
    - pricing_date = date of the market environment
    """
    def __init__(self, name, pricing_date): 
        self.name = name
        self.pricing_date = pricing_date
        # Define dictionaries that can be used to add constants (model params), lists (underlying, etc..) and yield curves (or other)
        self.constants = {}
        self.lists = {}
        self.curves = {}

    def add_constant(self, key, constant): 
        self.constants[key] = constant

    def get_constant(self, key): 
        return self.constants[key]

    def add_list(self, key, list_object): 
        self.lists[key] = list_object
    
    def get_list(self, key): 
        return self.lists[key]

    def add_curve(self, key, curve): 
        self.curves[key] = curve
        
    def get_curve(self, key): 
        return self.curves[key]

    def add_environment(self, env): # overwrites existing values, if they exist 
        """
        adds and overwrites whole market environments with constants, lists, and curves
        """
        self.constants.update(env.constants) 
        self.lists.update(env.lists) 
        self.curves.update(env.curves)

############ FUNCTION to GENERATE (PSEUDO) RANDOM NUMBERS ############
def sn_random_numbers(shape, antithetic=True, moment_matching=True, fixed_seed=False):
    """
    Returns an ndarray object of shape shape with (pseudo)random numbers that are standard normally distributed.
    INPUTS:
    - shape = generation of array with shape (o, n, m)
    - antithetic = generation of antithetic variates
    - moment_matching = matching of first and second moments
    - fixed_seed = flag to fix the seed
    OUTPUT:
    array of pseudo random number
    """
    if fixed_seed:
        np.random.seed(1000) 
    if antithetic:
        ran = np.random.standard_normal((shape[0], shape[1], shape[2] // 2))
        ran = np.concatenate((ran, -ran), axis=2) 
    else:
        ran = np.random.standard_normal(shape) 
    if moment_matching:
        ran = ran - np.mean(ran)
        ran = ran / np.std(ran) 
    if shape[0] == 1:
        return ran[0] 
    else:
        return ran

class simulation_class(object):
    """
    Providing base methods for simulation classes.
    ATTRIBUTES:
    - name = name of the object
    - mar_env =  market environment data for simulation (from class defined above)
    - corr = True if correlated with other model object
    """
    def __init__(self, name, mar_env, corr):
        self.name = name
        self.pricing_date = mar_env.pricing_date 
        # Get data from the market environment class
        self.initial_value = mar_env.get_constant('initial_value') 
        self.volatility = mar_env.get_constant('volatility') 
        self.final_date = mar_env.get_constant('final_date') 
        self.currency = mar_env.get_constant('currency') 
        self.frequency = mar_env.get_constant('frequency')
        self.paths = mar_env.get_constant('paths') 
        self.discount_curve = mar_env.get_curve('discount_curve') 
        try:
            # if time_grid in mar_env take that object (for portfolio valuation)
            self.time_grid = mar_env.get_list('time_grid')
        except:
            self.time_grid = None
        try:
            # if there are special dates, then add these 
            self.special_dates = mar_env.get_list('special_dates')
        except:
            self.special_dates = []
        self.instrument_values = None 
        self.correlated = corr
        if corr is True:
            # only needed in a portfolio context when risk factors are correlated
            self.cholesky_matrix = mar_env.get_list('cholesky_matrix') 
            self.rn_set = mar_env.get_list('rn_set')[self.name] 
            self.random_numbers = mar_env.get_list('random_numbers')
    
    def generate_time_grid(self):
        """
        Create the time grid for simulation
        """
        start = self.pricing_date
        end = self.final_date
        # pandas date_range function
        # freq = 'B' for Business Day, 'W' for Weekly, 'M' for Monthly
        time_grid = pd.date_range(start=start, end=end, freq=self.frequency).to_pydatetime()
        time_grid = list(time_grid)
            # enhance time_grid by start, end, and special_dates
        if start not in time_grid: 
            time_grid.insert(0, start)
            # insert start date if not in list
        if end not in time_grid: 
            time_grid.append(end)
            # insert end date if not in list
        if len(self.special_dates) > 0:
            # add all special dates 
            time_grid.extend(self.special_dates) # delete duplicates
            time_grid = list(set(time_grid)) # sort list
            time_grid.sort()
        self.time_grid = np.array(time_grid)
    
    def get_instrument_values(self, fixed_seed=True): 
        """
        Returns the current instrument values (array)
        """
        if self.instrument_values is None:
            # only initiate simulation if there are no instrument values
            self.generate_paths(fixed_seed=fixed_seed, day_count=365.) 
        elif fixed_seed is False:
            # also initiate resimulation when fixed_seed is False
            self.generate_paths(fixed_seed=fixed_seed, day_count=365.) 
        return self.instrument_values

class geometric_brownian_motion(simulation_class):
    """
    Class to generate simulated paths based on the Black-Scholes-Merton geometric Brownian motion model.
    ATTRIBUTES:
    - name = name of the object
    - mar_env = market environment data for simulation
    - corr = True if correlated with other model simulation object
    """
    def __init__(self, name, mar_env, corr=False): 
        super(geometric_brownian_motion, self).__init__(name, mar_env, corr)

    def update(self, initial_value=None, volatility=None, final_date=None): 
        """
        updates parameters
        """
        if initial_value is not None:
            self.initial_value = initial_value 
        if volatility is not None:
            self.volatility = volatility 
        if final_date is not None:
            self.final_date = final_date
        self.instrument_values = None

    def generate_paths(self, fixed_seed=False, day_count=365.): 
        """
        returns Monte Carlo paths given the market environment
        """
        if self.time_grid is None:  # method from generic simulation class
            self.generate_time_grid()
        # number of dates for time grid
        M = len(self.time_grid)
        # number of paths
        I = self.paths
        # ndarray initialization for path simulation 
        paths = np.zeros((M, I))
        # initialize first date with initial_value 
        paths[0] = self.initial_value
        if not self.correlated:
            # if not correlated, generate random numbers
            rand = sn_random_numbers((1, M, I), fixed_seed=fixed_seed)
        else:
            # if correlated, use random number object as provided # in market environment
            rand = self.random_numbers
        short_rate = self.discount_curve.short_rate # get short rate for drift of process
        for t in range(1, len(self.time_grid)):
            # select the right time slice from the relevant # random number set
            if not self.correlated:
                ran = rand[t] 
            else:
                ran = np.dot(self.cholesky_matrix, rand[:, t, :])
                ran = ran[self.rn_set]
            dt = (self.time_grid[t] - self.time_grid[t - 1]).days / day_count # difference between two dates as year fraction
            paths[t] = paths[t - 1] * np.exp((short_rate - 0.5 * self.volatility ** 2) * dt + self.volatility * np.sqrt(dt) * ran) 
            # generate simulated values for the respective date
        self.instrument_values = paths
    
class jump_diffusion(simulation_class):
    """
    Class to generate simulated paths based on the Merton (1976) jump diffusion model.
    ATTRIBUTES (as brownian motion)
    """
    def __init__(self, name, mar_env, corr=False): 
        super(jump_diffusion, self).__init__(name, mar_env, corr) 
        # additional parameters needed
        self.lamb = mar_env.get_constant('lambda')   # lambda = coefficient for the drift correction
        self.mu = mar_env.get_constant('mu')         # mu = avg return for the drift correction
        self.delt = mar_env.get_constant('delta')    # delta = volatility for the drift correction

    def update(self, initial_value=None, volatility=None, lamb=None, mu=None, delta=None, final_date=None):
        if initial_value is not None: 
            self.initial_value = initial_value
        if volatility is not None: 
            self.volatility = volatility
        if lamb is not None: 
            self.lamb = lamb
        if mu is not None: 
            self.mu = mu
        if delta is not None: 
            self.delt = delta
        if final_date is not None: 
            self.final_date = final_date
        self.instrument_values = None
    
    def generate_paths(self, fixed_seed=False, day_count=365.): 
        if self.time_grid is None:
            # method from generic simulation class
            self.generate_time_grid()
        M = len(self.time_grid)         # number of dates for time grid
        I = self.paths                  # number of paths
        paths = np.zeros((M, I))        # ndarray initialization for path simulation 
        paths[0] = self.initial_value   # initialize first date with initial_value 
        if self.correlated is False:
            # if not correlated, generate random numbers
            sn1 = sn_random_numbers((1, M, I), fixed_seed=fixed_seed)
        else:
            # if correlated, use random number object as provided in market environment
            sn1 = self.random_numbers
        # standard normally distributed pseudo-random numbers # for the jump component
        sn2 = sn_random_numbers((1, M, I), fixed_seed=fixed_seed)
        rj = self.lamb * (np.exp(self.mu + 0.5 * self.delt ** 2) - 1)
        short_rate = self.discount_curve.short_rate 
        for t in range(1, len(self.time_grid)):
            if self.correlated is False:
                ran = sn1[t] 
            else:
                ran = np.dot(self.cholesky_matrix, sn1[:, t, :])
                ran = ran[self.rn_set]
            dt = (self.time_grid[t] - self.time_grid[t - 1]).days / day_count
            poi = np.random.poisson(self.lamb * dt, I)
            # Poisson-distributed pseudo-random numbers for jump component 
            paths[t] = paths[t - 1] * (np.exp((short_rate - rj - 0.5 * self.volatility ** 2) * dt + self.volatility * np.sqrt(dt) * ran) + (np.exp(self.mu + self.delt * sn2[t]) - 1) * poi)
        self.instrument_values = paths

############ DERIVATIVES VALUATION ############
class valuation_class(object):
    """
    Basic class for single-factor valuation.
    ATTRIBUTES
    - name = name of the object
    - underlying = object modeling the single risk factor
    - mar_env = market environment data for valuation
    - payoff_func = derivatives payoff in oython syntax !!!!!!
    """
    def __init__(self, name, underlying, mar_env, payoff_func=''): 
        self.name = name
        self.pricing_date = mar_env.pricing_date
        try:
            # strike is optional
            self.strike = mar_env.get_constant('strike') 
        except:
            pass
        self.maturity = mar_env.get_constant('maturity')
        self.currency = mar_env.get_constant('currency')
        # simulation parameters and discount curve from simulation object self.frequency = underlying.frequency
        self.paths = underlying.paths
        self.discount_curve = underlying.discount_curve
        self.payoff_func = payoff_func
        self.underlying = underlying
        # provide pricing_date and maturity to underlying
        self.underlying.special_dates.extend([self.pricing_date,self.maturity])

    def update(self, initial_value=None, volatility=None, strike=None, maturity=None):
        """
        updates selected valuation parameters
        """
        if initial_value is not None: 
            self.underlying.update(initial_value=initial_value)
        if volatility is not None: 
            self.underlying.update(volatility=volatility)
        if strike is not None: 
            self.strike = strike 
        if maturity is not None:
            self.maturity = maturity
            # add new maturity date if not in time_grid 
            if maturity not in self.underlying.time_grid:
                self.underlying.special_dates.append(maturity)
                self.underlying.instrument_values = None
    
    def delta(self, interval=None, accuracy=4):
        if interval is None:
            interval = self.underlying.initial_value / 50.
        # forward-difference approximation
        # calculate left value for numerical delta
        value_left = self.present_value(fixed_seed=True)
        # numerical underlying value for right value
        initial_del = self.underlying.initial_value + interval 
        self.underlying.update(initial_value=initial_del)
        # calculate right value for numerical delta
        value_right = self.present_value(fixed_seed=True)
        # reset the initial_value of the simulation object 
        self.underlying.update(initial_value=initial_del - interval) 
        delta = (value_right - value_left) / interval
        # correct for potential numerical errors
        if delta < -1.0:
            return -1.0 
        elif delta > 1.0:
            return 1.0 
        else:
            return round(delta, accuracy)
        
    def vega(self, interval=0.01, accuracy=4):
        if interval < self.underlying.volatility / 50.: 
            interval = self.underlying.volatility / 50.
        # forward-difference approximation
        # calculate the left value for numerical vega 
        value_left = self.present_value(fixed_seed=True) # numerical volatility value for right value 
        vola_del = self.underlying.volatility + interval # update the simulation object 
        self.underlying.update(volatility=vola_del)
        # calculate the right value for numerical vega 
        value_right = self.present_value(fixed_seed=True)
        # reset volatility value of simulation object
        self.underlying.update(volatility=vola_del - interval) 
        vega = (value_right - value_left) / interval
        return round(vega, accuracy)

class valuation_mcs_european(valuation_class):
    """ 
    Class to value European options with arbitrary payoff by single-factor Monte Carlo simulation.
    """
    def generate_payoff(self, fixed_seed=False):
        """
        fixed_seed = use same/fixed seed for valuation
        """
        try:
            # strike is optional 
            strike = self.strike
        except: 
            pass

        paths = self.underlying.get_instrument_values(fixed_seed=fixed_seed) 
        time_grid = self.underlying.time_grid
        try:
            time_index = np.where(time_grid == self.maturity)[0]
            time_index = int(time_index) 
        except:
            print('Maturity date not in time grid of underlying.') 
        # Values need for valueation given different types of payoff
        maturity_value = paths[time_index]
        # average value over whole path
        mean_value = np.mean(paths[:time_index], axis=1)
        # maximum value over whole path
        max_value = np.amax(paths[:time_index], axis=1)[-1]
        # minimum value over whole path
        min_value = np.amin(paths[:time_index], axis=1)[-1]
        try:
            payoff = eval(self.payoff_func)   # use payoff function (eval because it is given as a string)
            return payoff 
        except:
            print("Error evaluating payoff function, python syntax problem")

    def present_value(self, accuracy=6, fixed_seed=False, full=False): 
        """    
        accuracy = number of decimals in returned result
        fixed_seed = use same/fixed seed for valuation to always get same result
        full = return also full 1d array of present values
        """
        cash_flow = self.generate_payoff(fixed_seed=fixed_seed)
        discount_factor = self.discount_curve.get_discount_factors((self.pricing_date, self.maturity))[0, 1]
        result = discount_factor * np.sum(cash_flow) / len(cash_flow) 
        if full:
            return round(result, accuracy), discount_factor * cash_flow 
        else:
            return round(result, accuracy)