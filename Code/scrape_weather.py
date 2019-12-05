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
# driver = webdriver.Chrome('/Users/chenzichu/Desktop/Adwin Lo/GWU_classes/DATS_6103_DataMining/Class08_WebScraping/chromedriver')
# try:
#     driver.get("https://www.wunderground.com/history/daily/us/va/arlington-county/KDCA/date/2017-1-1")
#     sleep(20)
#     soup = BeautifulSoup(driver.page_source, 'html5lib')
# except TimeoutException as ex:
#    print(ex.Message)
#    webDriver.navigate().refresh()




#%%

def get_table(soup):
    tables = soup.findAll('table')
    tab = tables[2]
    daylist=[]
    for tr in tab.tbody.findAll('tr'):
        hourlist=[]
        for td in tr.findAll('td'):
            line=td.getText()
            line = re.sub('[\xa0]', '', line)
            hourlist.append(line)
        daylist.append(hourlist)
    return daylist



#%%
#scraper weather from 365 days
bicycle17=pd.read_csv('./DataSet/bicycle17.csv')
datelist=list(bicycle17['Date'])
datelist=np.array(datelist)
datelist=np.unique(datelist)
#datelist2=list(datelist[56:90])
#July = list(datelist[181:212])
#Aug = list(datelist[212:243])
#Sep = list(datelist[243:273])
#WeatherData = pd.DataFrame( columns=['date','time','temp','humidity','windspeed'])

#%%
frames=[]
for date in datelist:
    driver = webdriver.Chrome('/Users/yukaiqi/Downloads/chromedriver')
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

WeatherData = pd.concat(frames)



# %%
AugData = pd.read_csv("./DataSet/AugData.csv")
JulyData = pd.read_csv("./DataSet/JulyData.csv")
SepData = pd.read_csv("./DataSet/SepData.csv")
OctData = pd.read_csv("./DataSet/SepData.csv")
p1Data = pd.read_csv("./DataSet/weather17p1.csv")
p4Data = pd.read_csv("./DataSet/weather17p4.csv")

Weather = pd.concat([p1Data, JulyData, AugData, SepData, OctData, p4Data], axis = 0, ignore_index = True)


# %%
