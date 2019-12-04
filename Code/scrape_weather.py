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
testWeatherData = pd.DataFrame( columns=['date','time','temp','humidity','windspeed'])


def gettime(soup,k):
    time=soup.select('#inner-content > div.region-content-main > div.row > div:nth-child(5) > div:nth-child(1) > div > lib-city-history-observation > div > div.observation-table.ng-star-inserted > table > tbody > tr:nth-child('+str(k)+') > td.mat-cell.cdk-column-dateString.mat-column-dateString.mat-table-sticky.ng-star-inserted > span')
    return(re.split(r'<|>',str(time[0]))[2] if (len(time)==1) else "error")
    
def gettemp(soup,k):
    temp=soup.select('#inner-content > div.region-content-main > div.row > div:nth-child(5) > div:nth-child(1) > div > lib-city-history-observation > div > div.observation-table.ng-star-inserted > table > tbody > tr:nth-child('+str(k)+') > td.mat-cell.cdk-column-temperature.mat-column-temperature.ng-star-inserted > lib-display-unit > span > span.wu-value.wu-value-to')
    return(re.split(r'<|>',str(temp[0]))[2] if (len(temp)==1) else "error")
    
def gethum(soup,k):
    hum=soup.select('#inner-content > div.region-content-main > div.row > div:nth-child(5) > div:nth-child(1) > div > lib-city-history-observation > div > div.observation-table.ng-star-inserted > table > tbody > tr:nth-child('+str(k)+') > td.mat-cell.cdk-column-humidity.mat-column-humidity.ng-star-inserted > lib-display-unit > span > span.wu-value.wu-value-to')
    return(re.split(r'<|>',str(hum[0]))[2] if (len(hum)==1) else "error")

def getwind(soup,k):
    wind=soup.select('#inner-content > div.region-content-main > div.row > div:nth-child(5) > div:nth-child(1) > div > lib-city-history-observation > div > div.observation-table.ng-star-inserted > table > tbody > tr:nth-child('+str(k)+') > td.mat-cell.cdk-column-windSpeed.mat-column-windSpeed.ng-star-inserted > lib-display-unit > span > span.wu-value.wu-value-to')
    return(re.split(r'<|>',str(wind[0]))[2] if (len(wind)==1) else "error")

#%%
for k in range(1,24):
    time=gettime(soup,k)
    temp=gettemp(soup,k)
    hum=gethum(soup,k)
    wind=getwind(soup,k)
    testWeatherData = testWeatherData.append({'date':'2017-01-01','time':time,'temp':temp,'humidity':hum,'windspeed':wind}, ignore_index=True)




#%%
#scraper weather from 365 days
bicycle17=pd.read_csv('/Users/chenzichu/Desktop/ItDMProj/DataSet/bicycle17.csv')
datelist=list(bicycle17['Date'])
datelist=np.array(datelist)
datelist=np.unique(datelist)
datelist1=list(datelist[0:3])
WeatherData = pd.DataFrame( columns=['date','time','temp','humidity','windspeed'])

#%%
frames=[testWeatherData]
for date in datelist1:
    driver = webdriver.Chrome('/Users/chenzichu/Desktop/Adwin Lo/GWU_classes/DATS_6103_DataMining/Class08_WebScraping/chromedriver')
    url = "https://www.wunderground.com/history/daily/us/va/arlington-county/KDCA/date/" + date
    try:
        driver.get(url)
        sleep(10)
        soup = BeautifulSoup(driver.page_source, 'html5lib')
    except TimeoutException as ex:
        print(ex.Message)
        webDriver.navigate().refresh()
    for k in range(1,24):
        time=gettime(soup,k)
        temp=gettemp(soup,k)
        hum=gethum(soup,k)
        wind=getwind(soup,k)
        WeatherData = WeatherData.append({'date':date,'time':time,'temp':temp,'humidity':hum,'windspeed':wind}, ignore_index=True)
    driver.quit()



# %%
