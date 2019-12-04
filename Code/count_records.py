#%%
import pandas as pd
bicycle1=pd.read_csv('/Users/chenzichu/Desktop/ItDMProj/DataSet/2017-capitalbikeshare-tripdata/2017Q1-capitalbikeshare-tripdata.csv')
bicycle2=pd.read_csv('/Users/chenzichu/Desktop/ItDMProj/DataSet/2017-capitalbikeshare-tripdata/2017Q2-capitalbikeshare-tripdata.csv')
bicycle3=pd.read_csv('/Users/chenzichu/Desktop/ItDMProj/DataSet/2017-capitalbikeshare-tripdata/2017Q3-capitalbikeshare-tripdata.csv')
bicycle4=pd.read_csv('/Users/chenzichu/Desktop/ItDMProj/DataSet/2017-capitalbikeshare-tripdata/2017Q4-capitalbikeshare-tripdata.csv')

#%%
def group_bicycle(bicycle):
     bicycle.head(3)
     bicycle["hour"] = [t.hour for t in pd.DatetimeIndex(bicycle['Start date'])]
     bicycle["day"] = [t.dayofweek for t in pd.DatetimeIndex(bicycle['Start date'])]
     bicycle["month"] = [t.month for t in pd.DatetimeIndex(bicycle['Start date'])]
     bicycle['Date'] = [t.floor('D') for t in pd.DatetimeIndex(bicycle['Start date'])]
     bicycle=bicycle.drop(columns=['Duration','End date','Start station number','Start station','End station number','End station','Bike number','Member type'])
     bicycle_cleaned=bicycle.groupby(["Date", "hour"]).size().reset_index(name="cnt")
     return bicycle_cleaned
#%%
cleaned1=group_bicycle(bicycle1)
cleaned2=group_bicycle(bicycle2)
cleaned3=group_bicycle(bicycle3)
cleaned4=group_bicycle(bicycle4)

# %%
frames=[cleaned1,cleaned2,cleaned3,cleaned4]
bicycle17=pd.concat(frames)

# %%
bicycle17.to_csv('/Users/chenzichu/Desktop/ItDMProj/DataSet/bicycle17.csv')

# %%
