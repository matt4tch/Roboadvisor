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
        
    


    
        
  
