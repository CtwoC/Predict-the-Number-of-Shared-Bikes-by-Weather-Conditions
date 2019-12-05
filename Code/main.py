#%%
import edaFunction as eda
import pandas as pd

# load file
dc = pd.read_csv("../DataSet/hour.csv")
dc2017 = pd.read_csv("../DataSet/hour2017.csv")

# london = pd.read_csv("../DataSet/london_merged.csv")


#%%
# preprocess
dc = dc.rename(columns={'yr':'year', 'mnth':'month', 'hr':'hour', 'weekday':'day', 'weathersit':'weather', 'temp':'TF', 'atemp':'TFF', 'hum':'Humidity', 'windspeed':'WindSpeed'})
# london = london.rename(columns={'weather_code':'weather', 't1':'TF', 't2':'TFF', 'hum':'Humidity', 'wind_speed':'WindSpeed', 'is_holiday': 'holiday', 'is_weekend': 'weekend'})
# london["hour"] = [t.hour for t in pd.DatetimeIndex(london.timestamp)]
# london["day"] = [t.dayofweek for t in pd.DatetimeIndex(london.timestamp)]
# london["month"] = [t.month for t in pd.DatetimeIndex(london.timestamp)]
# london['year'] = [t.year for t in pd.DatetimeIndex(london.timestamp)]
# london['year'] = london['year'].map({2015:0, 2016:1})

dc2017['TF'] = dc2017['TF'].map(lambda x: x.rstrip('F'))
dc2017['Humidity'] = dc2017['TF'].map(lambda x: x.rstrip('%'))
dc2017['WindSpeed'] = dc2017['TF'].map(lambda x: x.rstrip('mph'))
dc2017['TF'] = dc2017['TF'].astype('float')
dc2017['Humidity'] = dc2017['TF'].astype('float')
dc2017['WindSpeed'] = dc2017['TF'].astype('float')
maxTF = max(dc2017.TF)
maxHM = max(dc2017.Humidity)
maxWS = max(dc2017.WindSpeed)
dc2017['TF'] = dc2017['TF'].map(lambda x: x / maxTF)
dc2017['Humidity'] = dc2017['Humidity'].map(lambda x: x / maxHM)
dc2017['WindSpeed'] = dc2017['WindSpeed'].map(lambda x: x / maxWS)




dcEDA = eda.edaFunction(dc)
dc2017EDA = eda.edaFunction(dc2017)
# LondonEDA = eda.edaFunction(london)


#%%
# check Info
dcEDA.ckInfo()
# LondonEDA.ckInfo()
#%%
# check NA
dcEDA.ckNA()
# LondonEDA.ckNA()
#%%
# Drop NA
dcEDA.dropNA()
# LondonEDA.dropNA()
#%%
# Season
dcEDA.seasonEDA()
# LondonEDA.seasonEDA()
#%%
# Holiday
dcEDA.holidayEDA()
# LondonEDA.holidayEDA()
#%%
# Workingday
dcEDA.workingdayEDA()
#%%
# Weekend
#LondonEDA.weekendEDA()
#%%
# Weather
dcEDA.weatherTypeEDA()
# LondonEDA.weatherTypeEDA()
#%%
# Hour
dcEDA.hourEDA()
# LondonEDA.hourEDA()
#%%
# Day
dcEDA.dayEDA()
# LondonEDA.dayEDA()
#%%
# Month
dcEDA.monthEDA()
# LondonEDA.monthEDA()
#%%
# Year
dcEDA.yearEDA()
# LondonEDA.yearEDA()
#%%
# Hists
dcEDA.hists()
# LondonEDA.hists()

#%%
dcEDA.register()
#%%
# corelationMatrix
dcEDA.corelationMatrix()
# LondonEDA.corelationMatrix()

# %%
dc2017EDA.hourEDA()

#%%
dc2017EDA.hists()
#%%
dc2017EDA.corelationMatrix()

#%%