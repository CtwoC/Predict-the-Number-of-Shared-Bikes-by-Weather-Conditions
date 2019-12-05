#%%
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from time import sleep
from selenium.common.exceptions import TimeoutException
import re

#%%
driver = webdriver.Chrome('/Users/chenzichu/Desktop/Adwin Lo/GWU_classes/DATS_6103_DataMining/Class08_WebScraping/chromedriver')
try:
    driver.get("https://www.wunderground.com/history/daily/us/va/arlington-county/KDCA/date/2017-1-1")
    sleep(20)
    soup = BeautifulSoup(driver.page_source, 'html5lib')
except TimeoutException as ex:
   print(ex.Message)
   webDriver.navigate().refresh()


#%%




#%%
testWeatherData = pd.DataFrame( columns=['date','time','temp','humidity','windspeed'])


def get_table(soup):
    tables = soup.findAll('table')
    tab = tables[2]
    weathertable=pd.DataFrame(columns=[])
    daylist=[]
    for tr in tab.tbody.findAll('tr'):
        hourlist=[]
        for td in tr.findAll('td'):
            line=td.getText()
            line = re.sub('[\xa0]', '', line)
            hourlist.append(line)
        daylist.append(hourlist)
    return daylist


dayweather = pd.DataFrame.from_records(daylist)




#%%
#scraper weather from 365 days
bicycle17=pd.read_csv('/Users/chenzichu/Desktop/ItDMProj/DataSet/bicycle17.csv')
datelist=list(bicycle17['Date'])
datelist=np.array(datelist)
datelist=np.unique(datelist)
datelist2=list(datelist[56:90])
#WeatherData = pd.DataFrame( columns=['date','time','temp','humidity','windspeed'])

#%%
#frames=[]
for date in datelist2:
    driver = webdriver.Chrome('/Users/chenzichu/Desktop/Adwin Lo/GWU_classes/DATS_6103_DataMining/Class08_WebScraping/chromedriver')
    url = "https://www.wunderground.com/history/daily/us/va/arlington-county/KDCA/date/" + date
    driver.get(url)
    sleep(10)
    soup = BeautifulSoup(driver.page_source, 'html5lib')
    driver.quit()
    daylist=get_table(soup)
    dayweather = pd.DataFrame.from_records(daylist)
    choose=dayweather.drop(columns=[2,4,6,7,8])
    choose.columns=['time','temp','hum','wind','weather']
    choose['date']=date
    frames.append(choose)
weatherframe=pd.concat(frames)



# %%
