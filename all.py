import requests
import pandas as pd

# collecting DMO gilt announcements from 2005 to present

url = 'https://www.dmo.gov.uk/publications/?offset=0&itemsPerPage=1000000&parentFilter=1433&childFilter=1433|1450&startMonth=1&startYear=2005&endMonth=7&endYear=2021'
html = requests.get(url).content
df_list = pd.read_html(html)
dates = df_list[-1]
dates.to_excel('gilt_announcements_20052021.xlsx')

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

spot_infla_2005 = pd.read_excel('Inflation_Daily_2005-2015.xlsx', '4')
spot_infla_2005 = spot_infla_2005.dropna()
spot_infla_pres = pd.read_excel('2016-present_infla.xlsx', '4')
spot_infla_pres = spot_infla_pres.dropna()

implied_infla = pd.concat([spot_infla_2005,spot_infla_pres])
implied_infla = implied_infla.dropna(axis = 1)
implied_infla.columns

implied_infla = implied_infla.set_index('Date')
# 5 year bonds
implied_infla[5].plot()

# 10 year bonds
implied_infla[15].plot()


implied_infla[25].plot()

pd.set_option('display.max_columns', None)

implied_infla = implied_infla.reset_index()
implied_infla = pd.merge(implied_infla, exchange_rates, on='Date')
implied_infla['Exchange Index'] = implied_infla['Exchange Index'].str.replace(r',', r'.').astype('float') 
implied_infla['Exchange Index'] = pd.to_numeric(implied_infla['Exchange Index'])


implied_infla = implied_infla.reset_index()
implied_infla = pd.merge(implied_infla, ftse, on='Date')
implied_infla['close'] = pd.to_numeric(implied_infla['close'])
implied_infla['close_pctchange'] = implied_infla['close'].pct_change()


import numpy as np
np.log10(implied_infla[5]).plot()
np.log10(implied_infla['Exchange Index']).plot()
np.log10(implied_infla['close']).plot()

dates = dates[dates['Date published'] <= '2021-03-31']
dates = dates[dates['Date published'] >= '2011-01-04']

dates['amount'] = pd.to_numeric(dates['amount'])

implied_infla['log_exchange'] = np.log10(implied_infla['Exchange Index'])
implied_infla['log_close'] = np.log10(implied_infla['close'])
implied_infla['log_5y'] = np.log10(implied_infla[5])

implied_infla.corr()

import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
from dateutil.relativedelta import relativedelta

dates['pre_period'] = dates['Date published'] - pd.DateOffset(days=20)
dates['post_period'] = dates['Date published'] + pd.DateOffset(days=3)
dates['datepublished1'] = dates['Date published'] + pd.DateOffset(days=1)

from pandas.tseries.offsets import BDay
dates.pre_period = dates.pre_period.map(lambda x : x + 0*BDay())
dates.post_period = dates.post_period.map(lambda x : x + 0*BDay())
dates.datepublished1 = dates.datepublished1.map(lambda x : x + 0*BDay())
dates['Date published'] = dates['Date published'].map(lambda x : x + 0*BDay())


dates = dates[dates['post_period'] <= '2021-03-31']
dates = dates[dates['pre_period'] >= '2011-01-04']

dates = dates.sort_values(by=['Date published'])

implied_infla = implied_infla.reset_index()
a = dates['Date published'].tolist()
b = dates['pre_period'].tolist()
c = dates['post_period'].tolist()
d = dates['datepublished1'].tolist()

implied_infla['Date'] = pd.to_datetime(implied_infla['Date'],infer_datetime_format=True)
implied_infla = implied_infla[implied_infla['Date'] <= '2021-03-31']
implied_infla = implied_infla[implied_infla['Date'] >= '2011-01-04']
implied_infla = implied_infla.set_index("Date")

implied_infla.columns = [str(col) + '_year' for col in implied_infla.columns]
implied_infla.columns

fiveyear = implied_infla[['5_year','15_year','Exchange Index_year', 'close_year']]

import numpy as np
import pandas as pd
from statsmodels.tsa.arima_process import ArmaProcess
from causalimpact import CausalImpact
import re
import warnings
warnings.filterwarnings('ignore')

# 5 year bonds

params = []
errors = []
list2 = []
data = fiveyear.copy()
data.columns = ['y', 'X1', 'X2', 'X3']

for x in range(0,len(dates)):
    
    pre_period = [b[x], a[x]]
    post_period = [d[x], c[x]]
    try:
        ci = CausalImpact(data, pre_period, post_period)
        
        r1 = re.search('tail-area probability p: (.+?)\nPosterior prob.', ci.summary())
        params.append(r1.group(1))
        
        list2.append(ci.summary())
    
        if x%100==0:
            print(x)
            
    except ValueError as err:
        params.append('NaN')
        errors.append(err)
        list2.append('error')
        pass
        
results_5year = dates[['Date published','pre_period', 'post_period','amount', 'index']]
results_5year['five year p-value'] = params

listtemp = []
for x in range(0, len(list2)):
    try:
        actual_average = re.search('\nActual[\t ]*(.+?)[\t ]', list2[x]).group(1)
        actual_cumulative = re.search('\nActual[\t ]*\d.\d*[\t ]*(.+?)\nPrediction', list2[x]).group(1)

        rel_average = re.search('\n\nRelative effect \(s.d.\)[\t ]*(.+?)\s', list2[x]).group(1)
        rel_cumulative = re.search('\n\nRelative effect \(s.d.\).*-.*\)[\t ]*(.+?)\s', list2[x]).group(1)

        abs_average = re.search('\nAbsolute effect \(s.d.\)[\t ]*(.+?)\s', list2[x]).group(1)
        abs_cumulative = re.search('\nAbsolute effect \(s.d.\).*-.*\)[\t ]*(.+?)\s', list2[x]).group(1)

        listtemp.append([actual_average,actual_cumulative,rel_average,rel_cumulative,abs_average,abs_cumulative])
    
    except:
        listtemp.append(['error','error','error','error','error','error'])

dtemp = pd.DataFrame.from_records(listtemp, columns =['actual_average','actual_cumulative','rel_average',
                                                      'rel_cumulative','abs_average','abs_cumulative'])

df_5year = pd.concat([results_5year.reset_index(drop=True),dtemp.reset_index(drop=True)], axis=1)
df_5year = df_5year[~df_5year['actual_cumulative'].str.contains("error")]
df_5year = df_5year.dropna(subset = ['five year p-value'])

df_5year['rel_cumulative'] = df_5year['rel_cumulative'].str.replace(r'%', r'').astype('float') / 100.0
df_5year['rel_average'] = df_5year['rel_average'].str.replace(r'%', r'').astype('float') / 100.0
df_5year['actual_cumulative'] = pd.to_numeric(df_5year['actual_cumulative'])
df_5year['actual_average'] = pd.to_numeric(df_5year['actual_average'])
df_5year['abs_cumulative'] = pd.to_numeric(df_5year['abs_cumulative'])
df_5year['abs_average'] = pd.to_numeric(df_5year['abs_average'])

df_5year['five year p-value'] = pd.to_numeric(df_5year['five year p-value'])
df_5year['sig'] = 1
df_5year.loc[df_5year['five year p-value'] > 0.05, 'sig'] = 0

print('mean of non-significant gilt announcements: ', df_5year[df_5year['sig'] == 0]['amount'].mean())
print('mean of significant gilt announcements: ', df_5year[df_5year['sig'] == 1]['amount'].mean())

import pickle

with open("list1finally.txt", "rb") as fp:   # Unpickling
    data = pickle.load(fp)

listdf = pd.DataFrame.from_records(data)

listdf.columns = ['Date published', 'avg_auction_value', 'supposed_auction_value_todate', 'total_sales_todate',
                             'total_auctions_planned','total_auctions_remaining','total_planned_sales','total_sales_remaining']
                             
                             
df1 = df_5year.merge(listdf, on='Date published')

df1['avg_auction_value'] = pd.to_numeric(df1['avg_auction_value'])
df1['amount'] = pd.to_numeric(df1['amount'])
df1['auctions_value_remaining'] = df1['total_sales_remaining']/df1['total_auctions_remaining']
df1['surprise'] = df1['amount'] - df1['auctions_value_remaining']

df1['amount_pct'] = df1['amount']/df1['auctions_value_remaining']
df1['amount_pct2'] = df1['amount']/df1['avg_auction_value']
df1['amount_pct3'] = df1['amount']/df1['total_sales_remaining']

df1['total_sales_to_date'] = df1['total_planned_sales'] - df1['total_sales_remaining']
df1['diff'] = df1['amount']/df1['auctions_value_remaining']
df1['diff2'] = df1['amount']/df1['avg_auction_value']
df1['diff3'] = df1['amount']/df1['total_sales_remaining']


df1_sig = df1[df1['sig'] == 1]

df1_sig['diff'] = df1_sig['amount']/df1_sig['auctions_value_remaining']
df1_sig['diff2'] = df1_sig['amount']/df1_sig['avg_auction_value']
df1_sig['diff3'] = df1_sig['amount']/df1_sig['total_sales_remaining']
df1_sig['total_sales_to_date'] = df1_sig['total_planned_sales'] - df1_sig['total_sales_remaining']

df1['rel_average_useful'] = (1 + df1_sig['rel_average'])
df1.loc[df1['sig'] ==0,'rel_average_useful'] = 0

nominals = df1[df1['index'] == 0]

from scipy import stats
x = nominals[nominals['sig'] == 0]['amount_pct3']
y = nominals[nominals['sig'] == 1]['amount_pct3']
stats.ttest_ind(x, y,equal_var=False)

import statsmodels.api as sm
import numpy as np

Y = df1_sig['abs_cumulative']
X = df1_sig[['surprise','amount','total_auctions_remaining']]
X = sm.add_constant(X)

model = sm.OLS(Y,X)
results = model.fit()
results.summary()
