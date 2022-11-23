from IPython.display import display, Math, Latex
import pandas as pd
import numpy as np
import numpy_financial as npf
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import date
from dateutil.relativedelta import relativedelta

tickers = pd.read_csv('Tickers_Example.csv', header=None)

# Constant Definitions
max_volume = 200000
min_trading_days = 20
start_date = date(2022, 1, 1)
end_date = date(2022, 10, 31)

valid_tickers = []
tickers_history = []

for i in range(0, len(tickers)):
    temp_start = start_date
    
    # ensure that the ith ticker corresponds to a listed stock
    is_valid = True
    try:
        ticker = yf.Ticker(tickers.iloc[i][0])
        ticker_currency = ticker.info['currency']
    except:
        print("{0} is an invalid or delisted ticker".format(tickers.iloc[i][0]))
        is_valid = False
        
    if is_valid:    
        monthly_hist = ticker.history(start=start_date, end=end_date, interval='1mo')
        # only include US listed stocks with average monthly volume greater than or equal to max_volume are selected
        if ticker_currency == 'USD' and monthly_hist['Volume'].mean() >= max_volume:
            daily_hist = ticker.history(start=start_date, end=end_date, interval='1d')

            # include only months with a number of trading days greater than or equal to min_trading_days days
            while temp_start <= end_date:
                month_length = len(daily_hist[(daily_hist.index >= str(temp_start)) & (daily_hist.index <= pd.to_datetime(temp_start + relativedelta(months=1)))])
                if month_length < min_trading_days:
                    daily_hist.drop(daily_hist[(daily_hist.index >= str(temp_start)) & (daily_hist.index <= pd.to_datetime(temp_start + relativedelta(months=1)))].index.values, axis=0, inplace=True)
                temp_start += relativedelta(months=1)

            valid_tickers.append(ticker)
            tickers_history.append(daily_hist)
        

# *** Workings of main algorithm and create_portfolio, but DOES NOT RUN:
# (N: # of available stocks) (CONST k: # of stocks to choose from N) (portfolios: list of portfolios of k stocks)
# (remaining_tickers: list of strings representing tickers left to choose from)
# : 

def create_portfolio(tickers_added, remaining_tickers, pw_index_of_tickers_added, num_tickers_added, tickers_needed):
    
    if num_tickers_added == tickers_needed:
        # need to return a list of tickers added AND the standard deviation of the corresponding index
        return [tickers_added, pw_index_of_tickers_added]
    else:
        # create a dataframe with the price weighted index and remaining tickers not added to the index so far
        daily_returns_new = daily_returns[remaining_tickers]
        daily_returns_new.insert(0, 'PWI', pw_index_of_tickers_added.pct_change())

        # find the stock with the highest correlation to add to the price weighted index
        correlation_df = daily_returns_new.corr()
        series_stocks = correlation_df.iloc[0].replace(1, np.nan)
        stock_to_add = series_stocks.idxmax()
        
        # add the stock to the price weighted index
        pw_index_of_tickers_added += closing_prices[stock_to_add] / closing_prices[stock_to_add][0]
        tickers_added.append(stock_to_add)
        num_tickers_added += 1
        
        # remove the ticker from the list of remaining tickers
        print(remaining_tickers)
        
        remaining_tickers.remove(stock_to_add)
        
        return create_portfolio(tickers_added, remaining_tickers, pw_index_of_tickers_added, num_tickers_added, tickers_needed)

    
def main(N, k, portfolios, remaining_tickers, daily_returns):
    if N == k:
        return False
        #get_weighting(listof_stocks, portfolios, k)
    else:
        # create a correlation matrix of all stocks in daily_returns dataframe
        correlation_df = daily_returns.corr()

        # replace correlation of 1 with Nan values
        correlation_df.replace(1, np.nan, inplace=True)

        # find the two stocks with max correlation
        series_stocks = correlation_df.max()
        first_stock = series_stocks.idxmax(axis=0, skipna=True)
        series_stocks.drop(first_stock, inplace=True)
        second_stock = series_stocks.idxmax(axis=0, skipna=True)
 
        # create price weighted index (setting the initial price of each stock to $1)
        price_weighted_index = closing_prices[first_stock] / closing_prices[first_stock][0] + closing_prices[second_stock] / closing_prices[second_stock][0]
    
        remaining_tickers.remove(first_stock)
        remaining_tickers.remove(second_stock)
        
        pf = create_portfolio([first_stock, second_stock], remaining_tickers, price_weighted_index, 2, 12)
        portfolios.append(pf)
      
        # REMOVE stock with the lowest standard deviation from the list of stocks
        stock_to_drop = daily_returns.std().idxmin()
        daily_returns.drop(stock_to_drop, axis=1, inplace=True)
        remaining_tickers.remove(stock_to_drop)
    
        return main(N-1, k, portfolios, remaining_tickers, daily_returns)
        
        
remaining_tickers = [valid_tickers[i].info['symbol'] for i in range(len(valid_tickers))]
main(len(valid_tickers), 12, [], remaining_tickers, daily_returns)
