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
bicycle17 = pd.read_csv("./DataSet/bicycle17.csv")
bicycle17.columns = ['drop', 'Date', 'hour', 'cnt']
bicycle17 = bicycle17.drop(columns = ['drop'])


# %%
def cleanDfTime(row):
    hour = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    thisamt = row["time"]
    if (thisamt.strip() == "12:52 AM"): return hour[0]
    elif (thisamt.strip() == "1:52 AM"): return hour[1]
    elif (thisamt.strip() == "2:52 AM"): return hour[2]
    elif (thisamt.strip() == "3:52 AM"): return hour[3]
    elif (thisamt.strip() == "4:52 AM"): return hour[4]
    elif (thisamt.strip() == "5:52 AM"): return hour[5]
    elif (thisamt.strip() == "6:52 AM"): return hour[6]
    elif (thisamt.strip() == "7:52 AM"): return hour[7]
    elif (thisamt.strip() == "8:52 AM"): return hour[8]
    elif (thisamt.strip() == "9:52 AM"): return hour[9]
    elif (thisamt.strip() == "10:52 AM"): return hour[10]
    elif (thisamt.strip() == "11:52 AM"): return hour[11]
    elif (thisamt.strip() == "12:52 PM"): return hour[12]
    elif (thisamt.strip() == "1:52 PM"): return hour[13]
    elif (thisamt.strip() == "2:52 PM"): return hour[14]
    elif (thisamt.strip() == "3:52 PM"): return hour[15]
    elif (thisamt.strip() == "4:52 PM"): return hour[16]
    elif (thisamt.strip() == "5:52 PM"): return hour[17]
    elif (thisamt.strip() == "6:52 PM"): return hour[18]
    elif (thisamt.strip() == "7:52 PM"): return hour[19]
    elif (thisamt.strip() == "8:52 PM"): return hour[20]
    elif (thisamt.strip() == "9:52 PM"): return hour[21]
    elif (thisamt.strip() == "10:52 PM"): return hour[22]
    elif (thisamt.strip() == "11:52 PM"): return hour[23]
    else: return np.nan

    return np.nan

#%%
Weather.columns = ['drop', 'time', 'TF', 'Humidity', 'WindSpeed', 'weather', 'Date']
Weather['hour'] = Weather.apply(cleanDfTime, axis = 1)
Weather = Weather.drop(columns = ['drop', 'time'])
Weather = Weather.dropna()
Weather['hour'] = Weather['hour'].astype('int')
# Weather.reset_index()

# %%
Final_data = Weather.merge(bicycle17, on = ['Date','hour'], how = 'left')
Final_data['cnt'] = Final_data['cnt'].fillna(0)
Final_data['cnt'] = Final_data['cnt'].astype('int')
# Final_data.to_csv("hour2017.csv")

# %%

