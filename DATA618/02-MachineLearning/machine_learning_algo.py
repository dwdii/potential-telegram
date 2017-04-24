"""
This is a template algorithm on Quantopian for you to adapt and fill in.
"""
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import AverageDollarVolume
from quantopian.pipeline.filters.morningstar import Q1500US
from sklearn.ensemble import RandomForestClassifier
from collections import deque
import numpy as np
 
def initialize(context):
    """
    Called once at the start of the algorithm.
    """   
    # RETURNS 1137374.4%
    # ALPHA    1.65
    # BETA    14.22
    # SHARPE   0.31
    # DRAWDOWN -123.8%
    context.security = [sid(8554), sid(5061), sid(24), sid(3149)] # set the security
                        # SPY, MSFT, AAPL, GE
    context.window_length = 4 # number of bars or edges in a single decision tree
    context.history_to_fit = 50 # number of days back to use for training
    context.buys = []
   
    # Use a random forest classifier
    context.classifier = []
    for i in range(0, len(context.security)):
        context.classifier.append(RandomForestClassifier(n_estimators=20, criterion="entropy"))
        context.classifier[i].fitted = False
        context.buys.append(0)

    #context.prediction = 0 # Stores most recent prediction
    
    # set_commission(commission.PerShare(cost=0.0075, min_trade_cost=1))
    
    # Record tracking variables at the end of each day.
    schedule_function(daily_close, date_rules.every_day(), time_rules.market_close())
    schedule_function(daily_open, 
                      date_rules.every_day(), 
                      time_rules.market_open())
         
def slidingWindow(sequence,winSize,step=1):
    """Returns a generator that will iterate through
    the defined chunks of input sequence.  Input sequence
    must be iterable.
    
    From: https://scipher.wordpress.com/2010/12/02/simple-sliding-window-iterator-in-python/
    """
 
    # Verify the inputs
    try: it = iter(sequence)
    except TypeError:
        raise Exception("**ERROR** sequence must be iterable.")
    if not ((type(winSize) == type(0)) and (type(step) == type(0))):
        raise Exception("**ERROR** type(winSize) and type(step) must be int.")
    if step > winSize:
        raise Exception("**ERROR** step must not be larger than winSize.")
    if winSize > len(sequence):
        raise Exception("**ERROR** winSize must not be larger than sequence length.")
 
    # Pre-compute number of chunks to emit
    numOfChunks = ((len(sequence)-winSize)/step)+1
 
    # Do the work
    for i in range(0,numOfChunks*step,step):
        yield sequence[i:i+winSize]
 
def daily_close(context, data):
    """
    At end of day, re-fit the classifiers based on the new 
    trailing days of history (daily sliding window)
    """
    context.avg_price_diff = []
    context.stdev_price_diff = []
    for i in range(0, len(context.security)):
        prices = data.history(context.security[i], "price", context.history_to_fit, "1d") 
        priceDiffs = np.diff(prices)
        context.avg_price_diff.append(np.mean(priceDiffs))
        context.stdev_price_diff.append(np.std(priceDiffs))
    
        xData = []
        yData = []
        size = context.window_length + 1
        yNdx = size - 1
        for d in slidingWindow(priceDiffs, size):
            xData.append(d[0:context.window_length])
            yData.append(d[yNdx])
    
        context.classifier[i] = RandomForestClassifier(n_estimators=20, criterion="entropy")
        context.classifier[i].fit(xData, yData)
        context.classifier[i].fitted = True
 
def daily_open(context,data):
    """
    Called every day at market open.
    """
    
    for i in range(0, len(context.security)):
        if context.classifier[i].fitted:
            prices = data.history(context.security[i], "price", context.window_length, "1d") 
            prediction = context.classifier[i].predict(prices) # predict using 1-9 
            log.info("[" + context.security[i].symbol + "]: prediction: " + str(prediction) + ", avg_price_diff: " + str(round(context.avg_price_diff[i], 5)))
                
            # If prediction is greater than the x standard deviations above mean 
            # price diff over past x period then buy 10% of portfolio value.
            # context.prediction > (context.avg_price_diff + (0 * context.stdev_price_diff))     and 
            # context.prediction < (context.avg_price_diff - (0 * context.stdev_price_diff)) and 
            if prediction > 0:
                order_percent(context.security[i], 0.075)
                buyflag = 1
            elif context.buys > 0 and prediction < (0 - (2 * context.stdev_price_diff[i])):
                order_percent(context.security[i], -0.075)
                buyflag = -1
            else:
                buyflag = 0
                
            context.buys[i] += buyflag
            if i <= 1:
                record(context.security[i].symbol + "-prediction", prediction,
                       context.security[i].symbol + "-buy", buyflag)
