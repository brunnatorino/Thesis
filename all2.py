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

implied_infla = implied_infla.set_index('Date')


np.log10(implied_infla[5]).plot()
np.log10(implied_infla['Exchange Index']).plot()
np.log10(implied_infla['close']).plot()

dates = dates[dates['Date published'] <= '2021-03-31']
dates = dates[dates['Date published'] >= '2011-01-04']

dates['amount'] = pd.to_numeric(dates['amount'])
#dates= dates[dates['amount']>= 1000]

implied_infla['log_exchange'] = np.log10(implied_infla['Exchange Index'])
implied_infla['log_close'] = np.log10(implied_infla['close'])
#implied_infla['log_5y'] = np.log10(implied_infla[5])
del implied_infla['index']

a = implied_infla.corr()
a.to_excel('correlation_matrix.xlsx')

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

year3 = implied_infla[['3_year','15_year','Exchange Index_year', 'close_year']]
year4 = implied_infla[['4_year','15_year','Exchange Index_year', 'close_year']]
year5 = implied_infla[['5_year','15_year','Exchange Index_year', 'close_year']]
year6 = implied_infla[['6_year','16_year','Exchange Index_year', 'close_year']]
year7 = implied_infla[['7_year','17_year','Exchange Index_year', 'close_year']]
year8 = implied_infla[['8_year','17_year','Exchange Index_year', 'close_year']]
year9 = implied_infla[['9_year','18_year','Exchange Index_year', 'close_year']]
year10 = implied_infla[['10_year','20_year','Exchange Index_year', 'close_year']]

list_years = [year3,year4,year5,year6,year7,year8,year8,year10]

import numpy as np
import pandas as pd
from statsmodels.tsa.arima_process import ArmaProcess
from causalimpact import CausalImpact
import re
import warnings
warnings.filterwarnings('ignore')

#res = dates[['Date published','pre_period', 'post_period','amount', 'index']]

for x in range(1, len(list_years)):
    print(list_years[x].columns[0])
    
    params = []
    errors = []
    list2 = []
    data = list_years[x].copy()
    data.columns = ['y', 'X1', 'X2', 'X3']
    
    for n in range(0,len(dates)):
        pre_period = [b[n], a[n]]
        post_period = [d[n], c[n]]
        try:
            ci = CausalImpact(data, pre_period, post_period)
            r1 = re.search('tail-area probability p: (.+?)\nPosterior prob.', ci.summary())
            params.append(r1.group(1))
            list2.append(ci.summary())

        except ValueError as err:
            params.append('NaN')
            errors.append(err)
            list2.append('error')
            pass
    
    try:
        print(len(params))
        print('sig numbers:', len([float(n) for n in params if float(n) <= 0.05]))
        print('non-sig numbers:', len(len([float(n) for n in params if float(n) > 0.05])))
        print(len(errors))
    except TypeError:
        print('weird')
    
    listtemp = []
    for num in range(0, len(list2)):
        try:
            actual_average = re.search('\nActual[\t ]*(.+?)[\t ]', list2[num]).group(1)
            actual_cumulative = re.search('\nActual[\t ]*\d.\d*[\t ]*(.+?)\nPrediction', list2[num]).group(1)

            rel_average = re.search('\n\nRelative effect \(s.d.\)[\t ]*(.+?)\s', list2[num]).group(1)
            rel_cumulative = re.search('\n\nRelative effect \(s.d.\).*-.*\)[\t ]*(.+?)\s', list2[num]).group(1)

            abs_average = re.search('\nAbsolute effect \(s.d.\)[\t ]*(.+?)\s', list2[num]).group(1)
            abs_cumulative = re.search('\nAbsolute effect \(s.d.\).*-.*\)[\t ]*(.+?)\s', list2[num]).group(1)

            listtemp.append([actual_average,actual_cumulative,rel_average,rel_cumulative,abs_average,abs_cumulative])
        
        except:
            listtemp.append(['error','error','error','error','error','error'])
            
    if x == 0:
        res['year3_p-value'] = params
        
        dtemp = pd.DataFrame.from_records(listtemp, columns =['actual_average','actual_cumulative','rel_average',
                                                              'rel_cumulative','abs_average','abs_cumulative'])
        dtemp.columns = [str(col) + 'year3' for col in dtemp.columns]
        
        res = pd.concat([res.reset_index(drop=True),dtemp.reset_index(drop=True)], axis=1)
        
    elif x == 1:
        res['year4_p-value'] = params
        dtemp = pd.DataFrame.from_records(listtemp, columns =['actual_average','actual_cumulative','rel_average',
                                                              'rel_cumulative','abs_average','abs_cumulative'])
        dtemp.columns = [str(col) + 'year4' for col in dtemp.columns]
        res = pd.concat([res.reset_index(drop=True),dtemp.reset_index(drop=True)], axis=1)
        
    elif x == 2:
        res['year5_p-value'] = params
        dtemp = pd.DataFrame.from_records(listtemp, columns =['actual_average','actual_cumulative','rel_average',
                                                              'rel_cumulative','abs_average','abs_cumulative'])
        dtemp.columns = [str(col) + 'year5' for col in dtemp.columns]
        res = pd.concat([res.reset_index(drop=True),dtemp.reset_index(drop=True)], axis=1)
    elif x == 3:
        res['year6_p-value'] = params
        dtemp = pd.DataFrame.from_records(listtemp, columns =['actual_average','actual_cumulative','rel_average',
                                                              'rel_cumulative','abs_average','abs_cumulative'])
        dtemp.columns = [str(col) + 'year6' for col in dtemp.columns]
        res = pd.concat([res.reset_index(drop=True),dtemp.reset_index(drop=True)], axis=1)
    elif x == 4:
        res['year7_p-value'] = params
        dtemp = pd.DataFrame.from_records(listtemp, columns =['actual_average','actual_cumulative','rel_average',
                                                              'rel_cumulative','abs_average','abs_cumulative'])
        dtemp.columns = [str(col) + 'year7' for col in dtemp.columns]
        res = pd.concat([res.reset_index(drop=True),dtemp.reset_index(drop=True)], axis=1)
    elif x == 5:
        res['year8_p-value'] = params
        dtemp = pd.DataFrame.from_records(listtemp, columns =['actual_average','actual_cumulative','rel_average',
                                                              'rel_cumulative','abs_average','abs_cumulative'])
        dtemp.columns = [str(col) + 'year8' for col in dtemp.columns]
        res = pd.concat([res.reset_index(drop=True),dtemp.reset_index(drop=True)], axis=1)
    elif x == 6:
        res['year9_p-value'] = params
        dtemp = pd.DataFrame.from_records(listtemp, columns =['actual_average','actual_cumulative','rel_average',
                                                              'rel_cumulative','abs_average','abs_cumulative'])
        dtemp.columns = [str(col) + 'year9' for col in dtemp.columns]
        res = pd.concat([res.reset_index(drop=True),dtemp.reset_index(drop=True)], axis=1)
    elif x == 7:
        res['year10_p-value'] = params
        dtemp = pd.DataFrame.from_records(listtemp, columns =['actual_average','actual_cumulative','rel_average',
                                                              'rel_cumulative','abs_average','abs_cumulative'])
        dtemp.columns = [str(col) + 'year10' for col in dtemp.columns]
        res = pd.concat([res.reset_index(drop=True),dtemp.reset_index(drop=True)], axis=1)
