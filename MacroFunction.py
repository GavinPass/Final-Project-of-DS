from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import requests
from dateutil.parser import parse



# Web scraping for Treasury Yield
def GrabRate(Year):
    
    req = requests.get("https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView",params={'type':'daily_treasury_yield_curve','field_tdr_date_value':str(Year)})
    bs = BeautifulSoup(req.content,'html.parser')

    result=bs.find('table',class_='usa-table views-table views-view-table cols-26 sticky-enabled')
    data=result.find_all('tr')

    rates = pd.DataFrame()
    dates = []
    year_2 = []
    year_5 = []
    year_7 = []
    year_10 = []
    year_30 = []

    for i in data[1:]:
        dates.append(parse(i.find_all('td')[0].text))
        year_2.append(float(i.find_all('td')[16].text))
        year_5.append(float(i.find_all('td')[18].text))
        year_7.append(float(i.find_all('td')[19].text))
        year_10.append(float(i.find_all('td')[20].text))
        year_30.append(float(i.find_all('td')[22].text))
    
    rates['2-year rate'] = year_2
    rates['5-year rate'] = year_5
    rates['7-year rate'] = year_7
    rates['10-year rate'] = year_10
    rates['30-year rate'] = year_30
    rates['10s-2s spread']=rates['10-year rate']-rates['2-year rate']
    rates.index= dates
    
    return rates

# Web Scraping for last Ten Years GDP, Inflation and Unemployment rate
def GrabMacro(Name):
    
    req = requests.get('https://www.multpl.com/'+Name+'/table/by-month')
    bs = BeautifulSoup(req.content,'lxml')
    search = bs.find(id='datatable').find_all('tr')

    data=[]
    count=[]
    #Last ten years monthly data
    for i in search[1:121]:
    
        count.append(parse(i.find_all('td')[0].text))
        a=i.find_all('td')[1].text
        cleaned_string=''.join(filter(str.isdigit,a))
        float_value=float(cleaned_string)/10000
        data.append(float_value)

    return count,data
