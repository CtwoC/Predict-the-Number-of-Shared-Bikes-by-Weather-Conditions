#%%
import edaFunction as eda
import pandas as pd

# load file
dc = pd.read_csv("../DataSet/hour.csv")
london = pd.read_csv("../DataSet/london_merged.csv")

# preprocess
dc = dc.rename(columns={'yr':'year', 'mnth':'month', 'hr':'hour', 'weekday':'day', 'weathersit':'weather', 'temp':'TF', 'atemp':'TFF', 'hum':'Humidity', 'windspeed':'WindSpeed'})
london = london.rename(columns={'weather_code':'weather', 't1':'TF', 't2':'TFF', 'hum':'Humidity', 'wind_speed':'WindSpeed', 'is_holiday': 'holiday', 'is_weekend': 'weekend'})
london["hour"] = [t.hour for t in pd.DatetimeIndex(london.timestamp)]
london["day"] = [t.dayofweek for t in pd.DatetimeIndex(london.timestamp)]
london["month"] = [t.month for t in pd.DatetimeIndex(london.timestamp)]
london['year'] = [t.year for t in pd.DatetimeIndex(london.timestamp)]
london['year'] = london['year'].map({2015:0, 2016:1})

dcEDA = eda.edaFunction(dc)
LondonEDA = eda.edaFunction(london)


#%%
# check Info
dcEDA.ckInfo()
#%%
LondonEDA.ckInfo()
#%%
# check NA
dcEDA.ckNA()
#%%
LondonEDA.ckNA()
#%%
# Drop NA
dcEDA.dropNA()
#%%
LondonEDA.dropNA()
#%%
# Season
dcEDA.seasonEDA()
#%%
LondonEDA.seasonEDA()
#%%
# Holiday
dcEDA.holidayEDA()
#%%
LondonEDA.holidayEDA()
#%%
# Workingday
dcEDA.workingdayEDA()
#%%
# Weekend
LondonEDA.weekendEDA()
#%%
# Weather
dcEDA.weatherTypeEDA()
#%%
LondonEDA.weatherTypeEDA()
#%%
# Hour
dcEDA.hourEDA()
#%%
LondonEDA.hourEDA()
#%%
# Day
dcEDA.dayEDA()
#%%
LondonEDA.dayEDA()
#%%
# Month
dcEDA.monthEDA()
#%%
LondonEDA.monthEDA()
#%%
# Year
dcEDA.yearEDA()
#%%
LondonEDA.yearEDA()
#%%
# Hists
dcEDA.hists()
#%%
LondonEDA.hists()
#%%
# corelationMatrix
dcEDA.corelationMatrix()
#%%
LondonEDA.corelationMatrix()




# %%
