import requests
import pandas as pd

# collecting DMO gilt announcements from 2005 to present

url = 'https://www.dmo.gov.uk/publications/?offset=0&itemsPerPage=1000000&parentFilter=1433&childFilter=1433|1450&startMonth=1&startYear=2005&endMonth=7&endYear=2021'
html = requests.get(url).content
df_list = pd.read_html(html)
dates = df_list[-1]
dates.to_excel('gilt_announcements_20052021.xlsx')
