from IPython.display import display, Math, Latex

import pandas as pd
import numpy as np
import numpy_financial as npf
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import date
from dateutil.relativedelta import relativedelta
from collections import Counter

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
        
        else:
            print("{0} does not meet the required stock denomination or volume requirement".format(tickers.iloc[i][0]))
    
# initialize variables
closing_prices = pd.DataFrame()
daily_returns = pd.DataFrame()
expected_return = []
industries = []

# load yfinance data into DataFrames and lists
for i in range(len(valid_tickers)):
    closing_prices[valid_tickers[i].info['symbol']] = tickers_history[i]['Close']
    daily_returns[valid_tickers[i].info['symbol']] = closing_prices[valid_tickers[i].info['symbol']].pct_change()
    expected_return.append(daily_returns.iloc[i].mean() * 100)
    
# number of stocks to put into the portfolio
n = 12
   
# create_portfolio generates a portfolio of stocks by adding the stock most correlated to an already formed portfolio of at least one stock
# until tickers_needed number of stocks is reached

# create_portfolio: list list DataFrame Nat Nat DataFrame
def create_portfolio(tickers_added, remaining_tickers, pw_index_of_tickers_added, num_tickers_added, tickers_needed, daily_returns):
    
    if num_tickers_added == tickers_needed:
        # need to return a list of tickers added AND the price weighted index of the evenly distributed portfolio
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
        remaining_tickers.remove(stock_to_add)
        
        # recurse on the remaining available tickers
        return create_portfolio(tickers_added, remaining_tickers, pw_index_of_tickers_added, num_tickers_added, tickers_needed, daily_returns)

    
def main(N, k, portfolios, remaining_tickers, daily_returns):
    if N == k - 1:
        pf_stds = []
        for i in range(len(portfolios)):
            # append the standard deviations of each portfolio to a list
            pf_stds.append(portfolios[i][1].pct_change().std())
        
        # get maximum standard deviation portfolio
        loc = pf_stds.index(max(pf_stds))
        
        return portfolios[loc][0]
    else:
        # create a correlation matrix of all stocks in daily_returns DataFrame
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
        
        temp_tickers = remaining_tickers.copy()
        
        temp_tickers.remove(first_stock)
        temp_tickers.remove(second_stock)
        
        pf = create_portfolio([first_stock, second_stock], temp_tickers, price_weighted_index, 2, 12, daily_returns)

        portfolios.append(pf)

        # remove the stock with the lowest standard deviation from the list of stocks
        stock_to_drop = daily_returns.std().idxmin()
        
        # drop from both the dataframe AND the list of ticker names
        daily_returns.drop(stock_to_drop, axis=1, inplace=True)

        remaining_tickers.remove(stock_to_drop)
        
        # recurse on the remaining_tickers, having added one of them to the portfolio, getting closer to the base case
        return main(N-1, k, portfolios, remaining_tickers, daily_returns)
        
# create a list of remaining tickers stored as a list of strings         
remaining_tickers = [valid_tickers[i].info['symbol'] for i in range(len(valid_tickers))]

# retrieve a list of possible portfolios with the highest correlation
p = main(len(valid_tickers), n, [], remaining_tickers, daily_returns)

# initializing boundaries 
lower = 1/(2*n)
upper = 0.25

# number of test weights
trials = 1000
weights = []
ticker_names = [valid_tickers[i].info['symbol'] for i in range(len(valid_tickers))]

# generate "trials" amount of test weights
for i in range(trials):
    valid_weight = False
    while not valid_weight:
        temp = np.array(np.random.random(n))
        temp /= np.sum(temp)
        
        # ensure the generated test weights are within the given interval
        if np.all(temp >= lower) and np.all(temp <= upper):
            valid_weight = True
            weights.append(temp)

# create an equally weighted portfolio with n stocks where each stock has an initial value of $1 in the portfolio
pf = pd.DataFrame()

for i in range(n):
    pf[p[i]] = closing_prices[p[i]] / closing_prices[p[i]].iloc[0]
    
pf['Portfolio'] = pf.sum(axis=1)


# display the portfolio
plt.figure(figsize=(13,9))
plt.plot(pf.index, pf['Portfolio'] / n, marker='.', label='Equal Weighting') # division by n to scale the portfolio down

plt.title("Portfolio Value")
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.ylabel("Price (USD)")
plt.legend()

pf_diff = []
for w in weights:
    pf_temp = pd.DataFrame()
    
    # create portfolio of the n stocks in p, with a weighting of val assigned to each (i+1)th stock
    i = 0
    for val in w:
        pf_temp[p[i]] = pf[p[i]] * val
        i += 1
        
    pf_temp['Portfolio'] = pf_temp.sum(axis=1)
    
    # append the absolute difference between the start and end price to a related list
    pf_diff.append(abs(pf_temp['Portfolio'].iloc[0] - pf_temp['Portfolio'].iloc[-1]))
    
    # plot portfolio p with weightings corresponding to w
    x = pf_temp.index
    y = pf_temp['Portfolio']
    plt.plot(x, y, marker='.', label=str(num))

optimal_weight = weights[np.argmax(pf_diff)]

investment = 500000
closing = []
shares = []
value = []
for i in range(n):
    closing_price = valid_tickers[ticker_names.index(p[i])].history(start='2022-11-23', end='2022-11-24')['Close']
    closing.append(np.float32(closing_price)[0])
    shares.append(investment*optimal_weight[i] / closing[i])
    value.append(closing[i]*shares[i])

Portfolio_Final = pd.DataFrame(index=[i for i in range(1, n+1)])

Portfolio_Final['Ticker'] = p
Portfolio_Final['Price'] = closing
Portfolio_Final['Shares'] = shares
Portfolio_Final['Value'] = value
Portfolio_Final['Weight'] = optimal_weight * 100

# check to ensure that the weights adds to $100 and the investment size is correct
print("Total Weight: {0}%".format(sum(Portfolio_Final['Weight'])))
print("Total Investment Value: ${0}".format(sum(Portfolio_Final['Value'])))

display(Portfolio_Final)

Stocks_Final = Portfolio_Final[['Ticker', 'Shares']]
Stocks_Final.to_csv("Stocks_Group_13.csv")
display(Stocks_Final)
  
