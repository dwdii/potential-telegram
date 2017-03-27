"""
This is a template algorithm on Quantopian for you to adapt and fill in.
"""
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import AverageDollarVolume
from quantopian.pipeline.filters.morningstar import Q1500US
from statsmodels.tsa import stattools  

ENABLED = 2
SPREADMEAN = 3
SPREADSTDEV = 4
INTRADE = 5

 
def initialize(context):
    """
    Called once at the start of the algorithm.
    """   
    # Rebalance every day, 1 hour after market open.
    #schedule_function(my_rebalance, date_rules.every_day(), time_rules.market_open(hours=1))
     
    # Record tracking variables at the end of each day.
    schedule_function(my_record_vars, date_rules.every_day(), time_rules.market_close())
     
    # Create our dynamic stock selector.
    #attach_pipeline(make_pipeline(), 'my_pipeline')
    
    context.thePairs = [ [sid(4283),  # 'KO', 
                          sid(5885),  #'PEP'], 
                          False, None, None, False], 
                         [sid(8229),  # 'WMT', 
                          sid(21090), #'TGT'],
                          False, None, None, False],
                         [sid(8347),    # 'XOM', 
                          sid(23112), #CVX
                          False, None, None, False]
                       ]   
         

def before_trading_start(context, data):
    """
    Called every day before market open.
    """
    pass
     
def my_assign_weights(context, data):
    """
    Assign weights to securities that we want to order.
    """
    pass
 
def my_record_vars(context, data):
    """
    Plot variables at the end of each day.
    """
    for pair in context.thePairs:
        #log.info("Working: " + pair[0].symbol + ", " + pair[1].symbol)
        data1 = data.history(pair[0], 'close', 30, '1d')
        data2 = data.history(pair[1], 'close', 30, '1d')
        pair_s = is_pair_stationary(data1, data2)
        if(pair_s[0] and pair_s[1]):
            # The Null hypothesis is that there is no cointegration, 
            # the alternative hypothesis is that there is cointegrating
            # relationship. If the pvalue is small, below a critical size, 
            # then we can reject the hypothesis that there is no 
            # cointegrating relationship.
            result = stattools.coint(data1, data2)
            if(result[1] < 0.05):
                log.info("Cointegrated: " + pair[0].symbol + ", " + pair[1].symbol)
                log.info("coint: " + str(result)) 
                spread = data1 - data2
                u = spread.mean()
                sd = spread.std()
                log.info("spread.mean: " + str(u)) 
                log.info("spread.stdev: " + str(sd)) 
                pair[ENABLED] = True
                pair[SPREADMEAN] = u
                pair[SPREADSTDEV] = sd
 
def is_pair_stationary(data1, data2, verbose = False):
    
    #log.info(str(data1)) 
    ad1 = stattools.adfuller(data1)

    #log.info(str(data2)) 
    ad2 = stattools.adfuller(data2)

    if verbose:
        print [ad1, ad2]
        
    s1 = ad1[1] < 0.05 
    s2 = ad2[1] < 0.05
    if(s1 and s2 and verbose):
        print "Both are stationary"
    
    return [s1, s2]
       
def handle_data(context,data):
    """
    Called every minute.
    """
    for pair in context.thePairs:
        if(pair[ENABLED]):
            if data.can_trade(pair[1]):
                p1 = data.current(pair[0], 'price')
                p2 = data.current(pair[1], 'price')
                spread = p1 - p2                
                amount = 1000 / p2
                if not pair[INTRADE]:
                    if spread >= (pair[SPREADMEAN] + (2 * pair[SPREADSTDEV])):
                        order(pair[1], amount)
                elif pair[INTRADE]:
                    if spread < (pair[SPREADMEAN] + (2 * pair[SPREADSTDEV])):
                        order(pair[1], -1 * amount)
                        #pair[ENABLED] = False
                        pair[INTRADE] = False
                         
            
    
                
        
        
                           
                           
                           
                           
                           
