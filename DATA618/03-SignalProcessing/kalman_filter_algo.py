"""
This is a template algorithm on Quantopian for you to adapt and fill in.
"""
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import AverageDollarVolume
from quantopian.pipeline.filters.morningstar import Q1500US
import numpy as np
import numpy.linalg as npla
 
def initialize(context):
    """
    Called once at the start of the algorithm.
    """   
    
    context.stocks = [sid(8554)] #[sid(16841)]
    
    # Per p48,49 of Rankin (Kalman filtering approach to market price forecasting)
    # Fk: state transition matrix (n x n)
    # Hk: connection vector (n x n)
    context.Fk = np.asarray([1, 0.1, 0, 0.1]).reshape((2,2))
    context.Hk =  np.eye(2) 
    context.Rk = np.eye(2) * 0.00000833 # 0.01^2 / 12 from equation 3-13
    
    context.xhat_k = None
    context.P_k = None
    context.Qk = None    
    
    context.buys = 0
    
    # Rebalance every day
    schedule_function(on_update, date_rules.every_day(), time_rules.market_close(hours=1))
     
    # Record tracking variables at the end of each day.
    #schedule_function(my_record_vars, date_rules.every_day(), time_rules.market_close(hours=1))
 
   

def on_update(context,data):
    """
    Execute orders according to our schedule_function() timing. 
    """
    sec = context.stocks[0]
    
    if context.xhat_k == None:
        # Initial scenario. lets setup our 
        histData = data.history(sec, "close", 50, "1d")
        
        muSec = np.mean(histData)
        sdSec = np.std(histData)

        diffSec = np.diff(histData)
        sdDiffSec = np.std(diffSec)

        muDiff = np.mean(diffSec)
        sdDiff = np.std(diffSec)
        
        diffSec2 = np.diff(diffSec)
        sdDiffSec2 = np.std(diffSec2)
        
        context.xhat_k = np.asarray([muSec, muDiff]).reshape((2,1))
        context.P_k =  np.asarray([pow(sdSec,2), 0, 0, pow(sdDiff, 2)]).reshape((2,2))
        # variance of differences
        context.Qk = np.asarray([pow(sdDiffSec,2), 0, 0, pow(sdDiffSec2, 2)]).reshape((2,2)) 
        
        log.info("Initial context.xhat_k=" + str(context.xhat_k))
        log.info("Initial context.P_k=" + str(context.P_k))
        log.info("Initial context.Qk=" + str(context.Qk))
        log.info("Initial context.Hk=" + str(context.Hk))
        log.info("Initial context.Fk=" + str(context.Fk))
        log.info("Initial context.Rk=" + str(context.Rk))
    else:
        # Run the Kalman filter to update our state and predict tomorrow.
        kalman_filter(context, data, sec)
        
        curr = data.current(sec, "price")
        changeRatio = context.xhat_k[0] / curr
        log.info("changeRatio=" + str(changeRatio))
        record(changeRatio=changeRatio)
        if(changeRatio > 1):
            # buy some - at least 5%, or more with a strong signal
            order_percent(sec, 0.10 * changeRatio)
        elif context.portfolio.positions[sec] > 0:
            # If we have positions, then sell some
            sellRatio = (1 - changeRatio)
            toSell = (-.01 * sellRatio)
            log.info("Sell: " + str(toSell))
            order_percent(sec, toSell)

def extra_info(context, data, sec):
        curr = data.current(sec, "price")
        log.info("curr=" + str(curr))
        
        hist = data.history(sec, "close", 2, "1d")
        #log.info("Hist: " + str(hist))
        
        diffNp = np.diff(hist)
        log.info(diffNp)
        
        #diff = curr - hist[0]
        #log.info(diff)
    
        # Build current position vector
        x = np.asarray([curr, diffNp[0]])
        log.info(x)       
    
        
        
def kalman_filter(context, data, sec):
    """
    References:
    
    Rankin (Kalman filtering approach to market price forecasting)
    http://lib.dr.iastate.edu/cgi/viewcontent.cgi?article=9290&context=rtd
    
    https://courses.cs.washington.edu/courses/cse466/11au/calendar/14-StateEstimation-posted.pdf
    
    https://dsp.stackexchange.com/questions/3255/kalman-filter-in-practice
    
    https://dsp.stackexchange.com/questions/2347/how-to-understand-kalman-gain-intuitively
    
    https://en.wikipedia.org/wiki/Kalman_filter
    
    """
    
    log.info("------------------------")
    
    P_k = context.P_k
    Qk = context.Qk

    Hk = context.Hk
    Fk = context.Fk
    Rk = context.Rk
        
    diff = np.diff(data.history(sec, "close", 2, "1d"))
    z_k = np.asarray([data.current(sec, "price"), diff]).reshape((2,1))
    log.info("z_k=" + str(z_k))

    #xhatk_k1 = Fk * context.xhat_k + Bkuk
    #log.info("xhatk_k1=" + str(xhatk_k1))

    # Update based on recent observation of actual current price z_k
    #
    # context.Kk = P_k * np.transpose(Hk) * npla.inv(Hk * P_k * np.transpose(Hk) + Rk) # 3-8
    # For scalar outputs, inverse is simply division
    # Also helpful: 
    # https://dsp.stackexchange.com/questions/2347/how-to-understand-kalman-gain-intuitively
    HkP_k = np.dot(Hk, P_k)
    log.info("HkP_k=" + str(HkP_k))

    Sk = np.dot(HkP_k, np.transpose(Hk)) + Rk
    #log.info("np.dot(HkP_k, np.transpose(Hk))=" + str(np.dot(HkP_k, np.transpose(Hk))))
    log.info("Sk=" + str(Sk))

    #log.info("np.dot(P_k, np.transpose(Hk))=" + str(np.dot(P_k, np.transpose(Hk))))
    context.Kk = np.dot(np.dot(P_k, np.transpose(Hk)), npla.inv(Sk)) # 3-8
    log.info("context.Kk=" + str(context.Kk))

    yk = z_k - (np.dot(Hk, context.xhat_k))
    log.info("yk=" + str(yk))

    xhatk = context.xhat_k + (np.dot(context.Kk, yk)) # 3-7
    log.info("xhatk=" + str(xhatk))

    Pk = (np.eye(2) - (context.Kk * Hk)) * P_k # 3-9
    log.info("Pk=" + str(Pk))


    # Predict tomorrow's price 
    context.xhat_k = np.dot(Fk, xhatk) # 3-10
    log.info("context.xhat_k=" + str(context.xhat_k ))

    context.P_k = Fk * Pk * np.transpose(Fk) + Qk # 3-11
    log.info("context.P_k=" + str(context.P_k))

    record(error=yk[0])

