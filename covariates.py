exchange_rates = pd.read_csv('exchangerates.xlsm - ObservationData (1).csv')
exchange_rates = exchange_rates.T
new_header = exchange_rates.iloc[0] #grab the first row for the header
exchange_rates = exchange_rates[1:] #take the data less the header row
exchange_rates.columns = new_header
exchange_rates = exchange_rates.reset_index()
exchange_rates['index'] = pd.to_datetime(exchange_rates['index'] )
exchange_rates['Date'] = exchange_rates['index'].copy()
del exchange_rates['index']


ftse = pd.read_csv('ftse100close.csv').T
ftse = ftse[1:]
ftse = ftse.reset_index()
ftse.columns = ['Date','close']
ftse['close'] = ftse['close'].str.replace(',','.')
ftse = ftse.dropna()
ftse['close'] = pd.to_numeric(ftse['close'])
ftse['Date'] = pd.to_datetime(ftse['Date'],infer_datetime_format=True)

# cleaning up dates dataset
dates = dates.dropna()
dates = dates[dates['Publication title'].str.contains("Treasury Gilt")]
dates = dates[~dates['Publication title'].str.contains("Result")]
dates['amount'] = dates['Publication title'].str.split('Auction of ').str[1]
dates['amount'] = dates['amount'].str.split('of').str[0]
dates.loc[dates['Publication title'].str.contains("Index"), 'index'] = 1
dates.loc[~dates['Publication title'].str.contains("Index"), 'index'] = 0
dates.amount.replace('\D+','',regex=True,inplace=True)
dates = dates.dropna()
dates['Date published'] = pd.to_datetime(dates['Date published'],infer_datetime_format=True)
