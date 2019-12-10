#%%
import pandas as pd
import os
from statsmodels.formula.api import ols

# load file
os.chdir("../ItDMProj/DataSet")
dc = pd.read_csv("hour.csv")

# preprocess
dc = dc.rename(columns={'yr':'year', 'mnth':'month', 'hr':'hour', 'weekday':'day', 'weathersit':'weather', 'temp':'TF', 'atemp':'TFF', 'hum':'Humidity', 'windspeed':'WindSpeed'})
dc.head()

#%%
# Create dummy variables for categorial type variable 'weather' and 'season'

Wtype_dummies = pd.get_dummies(dc['weather'], prefix='w')
dc= pd.concat([dc, Wtype_dummies], axis=1)
Season_dummies = pd.get_dummies(dc['season'], prefix='s')
dc= pd.concat([dc, Season_dummies], axis=1)

# %%
# this part is to compare temperature for each season using s_4 as the benchmark.

X = dc[['s_1', 's_2', 's_3']]
Y = dc['TF']
modelseason0 = ols(formula= 'Y ~ X' , data=dc).fit()
print(modelseason0.summary())

# we can see from the result that the rank of average temperature for four seasons from low to high is : s_1, s_4, s_2, s_3
# %%
# Using all variables showing in the data set and 'cnt' as the dependent variable
X = dc[['holiday', 'workingday', 'TF', 'TFF', 'Humidity', 'WindSpeed']]
Y = dc['cnt']
modeldc0 = ols(formula= 'Y ~ X + C(hour) + C(weather) + C(season)' , data=dc).fit()
print(modeldc0.summary())

#%%
# Part A:
# build a linear model for the number of Casual Users, and get rid of w_4 and s_4 to avoid multicolinearity

X = dc[['holiday', 'workingday', 'TF', 'Humidity', 'WindSpeed', 'w_1', 'w_2', 'w_3', 's_1', 's_2', 's_3']]
Y = dc['casual']
modeldcA = ols(formula= 'Y ~ X' , data=dc).fit()
print(modeldcA.summary())

# We can see from the result that the 'Weather Situation' and 'Windspeed' do not have a significant effect on the casual users renting.
# Also we need to run regression seperately for 'workingday' and 'holiday', since the definition of these two variables are overlapping in some way and will effect the result

# %%

X = dc[['holiday', 'TF', 'Humidity', 's_1', 's_2', 's_3']]
Y = dc['casual']
modeldcA1 = ols(formula= 'Y ~ X' , data=dc).fit()
print(modeldcA1.summary())

# We can see from the result that on holiday there are more casual users renting bike.
# Also, the higher temperature, more casual users will rent the bike.
# Since another multicolinearty occurs with 'season' and 'tempreture', we will seperate them for regression.
# %%
# the following four models are the situations for 'holiday' and 'workingday' when using 'TF' and 'season' seperately for regression
# and them do follow our commen sense that:
# 1. more casual users rent bike on holidays when temperaute is high and humidity is low
X = dc[['holiday', 'TF', 'Humidity']]
Y = dc['casual']
modeldcA1_1 = ols(formula= 'Y ~ X' , data=dc).fit()
print(modeldcA1_1.summary())
# %%
# 2. more casual users rent bike on holidays when humidity is low and season have higher rank in temperatue 
X = dc[['holiday', 'Humidity',  's_1', 's_2', 's_3']]
Y = dc['casual']
modeldcA1_2 = ols(formula= 'Y ~ X' , data=dc).fit()
print(modeldcA1_2.summary())
# %%
# 3. more casual users are less likely to rent bike on workingday 
# and if they do more casual users rent bike when temperaute is high and humidity is low
X = dc[['workingday', 'TF', 'Humidity']]
Y = dc['casual']
modeldcA2_1 = ols(formula= 'Y ~ X' , data=dc).fit()
print(modeldcA2_1.summary())

# %%
# 4. more casual users are less likely to rent bike on workingday 
# and if they do more casual users rent bike when humidity is low and season have higher rank in temperatue 
X = dc[['workingday', 'Humidity', 's_1', 's_2', 's_3']]
Y = dc['casual']
modeldcA2_2 = ols(formula= 'Y ~ X' , data=dc).fit()
print(modeldcA2_2.summary())

#%%
# Part B:
# build a linear model for the number of registered Users, and get rid of w_4 and s_4 to avoid multicolinearity

X = dc[['holiday', 'workingday', 'TF', 'Humidity', 'WindSpeed', 'w_1', 'w_2', 'w_3', 's_1', 's_2', 's_3']]
Y = dc['registered']
modeldcB = ols(formula= 'Y ~ X' , data=dc).fit()
print(modeldcB.summary())

# We can see from the result that the 'Weather Situation' and 'Windspeed' do not have a significant effect on the casual users renting.
# Also we need to run regression seperately for 'workingday' and 'holiday', since the definition of these two variables are overlapping in some way and will effect the result
# %%

X = dc[['holiday', 'TF', 'Humidity', 's_1', 's_2', 's_3']]
Y = dc['registered']
modeldcB1 = ols(formula= 'Y ~ X' , data=dc).fit()
print(modeldcB1.summary())

# We can see from the result that on holiday there are less registered users renting bike. 
# since for the registered user, the main reason they register is that they will use a lot and we can imagine those people are because of work commute need
# In this case, when it comes to holiday, those people will just rest and not using bikes
# Also, the higher temperature, more casual users will rent the bike.
# Since another multicolinearty occurs with 'season' and 'tempreture', we will seperate them for regression.

# %%
# the following four models are the situations for 'holiday' and 'workingday' when using 'TF' and 'season' seperately for regression
# and the results do follow our commen sense that:
# 1. less registered users rent bike on holidays 
# and if they do, more people will rent bike when temperaute is high and humidity is low
X = dc[['holiday', 'TF', 'Humidity']]
Y = dc['registered']
modeldcB1_1 = ols(formula= 'Y ~ X' , data=dc).fit()
print(modeldcB1_1.summary())
# %%
# 2. less registered users rent bike on holidays 
# and if they do, more people will rent bike when humidity is low
# the rank of the number of people renting bikes for each season from low to high is s_1, s_2, s_4, s_3
# This is different from the result we get from 'TF' (since model0 has the temperature order from low to high as s_1, s_4, s_2, s_3)
X = dc[['holiday', 'Humidity',  's_1', 's_2', 's_3']]
Y = dc['registered']
modeldcB1_2 = ols(formula= 'Y ~ X' , data=dc).fit()
print(modeldcB1_2.summary())
# %%
# 3. more registered users are renting bike on workingday when humidity is low and seasons have higher rank in temperatue 

X = dc[['workingday', 'TF', 'Humidity']]
Y = dc['registered']
modeldcB2_1 = ols(formula= 'Y ~ X' , data=dc).fit()
print(modeldcB2_1.summary())

# %%
# 4. more registered users are renting bike on workingday when humidity is low 
# the rank of the number of people renting bikes for each season from low to high is s_1, s_2, s_4, s_3
# This is different from the result we get from 'TF' (since model0 has the temperature order from low to high as s_1, s_4, s_2, s_3)

X = dc[['workingday', 'Humidity', 's_1', 's_2', 's_3']]
Y = dc['registered']
modeldcB2_2 = ols(formula= 'Y ~ X' , data=dc).fit()
print(modeldcB2_2.summary())

#%%
# Part C:
# build a linear model for the number of cnt Users(casual+registered), and get rid of w_4 and s_4 to avoid multicolinearity
# since the 'Weather Situation' and 'Windspeed' in both casual and registered users do not have a significant effect on the casual users renting
# we will get rid of them and also we will run regression seperately for 'workingday' and 'holiday', since the definition of these two variables are overlapping in some way and will effect the result
X = dc[['workingday', 'TF', 'Humidity', 's_1', 's_2', 's_3']]
Y = dc['cnt']
modeldcC = ols(formula= 'Y ~ X' , data=dc).fit()
print(modeldcC.summary())

# we can see from the result that more cnt users rent bike on workingday
# and more people will rent bike when humidity is low and temperatue is high
# Since another multicolinearty occurs with 'season' and 'tempreture', we will seperate them for regression.
# %%

X = dc[['holiday', 'TF', 'Humidity', 's_1', 's_2', 's_3']]
Y = dc['cnt']
modeldcC1 = ols(formula= 'Y ~ X' , data=dc).fit()
print(modeldcC1.summary())

# We can see from the result that less cnt users rent bike on holiday 
# and more people will rent bike when humidity is low and temperatue is high
# Since another multicolinearty occurs with 'season' and 'tempreture', we will seperate them for regression.
# %%
# the following four models are the situations for 'holiday' and 'workingday' when using 'TF' and 'season' seperately for regression
# and them do follow our commen sense that:
# 1. less cnt users rent bike on holiday when temperaute is high and humidity is low
X = dc[['holiday', 'TF', 'Humidity']]
Y = dc['cnt']
modeldcC1_1 = ols(formula= 'Y ~ X' , data=dc).fit()
print(modeldcC1_1.summary())
# %%
# 2. less cnt users rent bike on holiday when humidity is low 
# the rank of the number of people renting bikes for each season from low to high is s_1, s_2, s_4, s_3
# This is different from the result we get from 'TF' (since model0 has the temperature order from low to high as s_1, s_4, s_2, s_3)

X = dc[['holiday', 'Humidity',  's_1', 's_2', 's_3']]
Y = dc['cnt']
modeldcC1_2 = ols(formula= 'Y ~ X' , data=dc).fit()
print(modeldcC1_2.summary())
# %%
# 3. more cnt users rent bike on workingday when temperaute is high and humidity is low
X = dc[['workingday', 'TF', 'Humidity']]
Y = dc['cnt']
modeldcC2_1 = ols(formula= 'Y ~ X' , data=dc).fit()
print(modeldcC2_1.summary())

# %%
# 4. more cnt users rent bike on workingday when humidity is low 
# the rank of the number of people renting bikes for each season from low to high is s_1, s_2, s_4, s_3
# This is different from the result we get from 'TF' (since model0 has the temperature order from low to high as s_1, s_4, s_2, s_3)

X = dc[['workingday', 'Humidity', 's_1', 's_2', 's_3']]
Y = dc['cnt']
modeldcC2_2 = ols(formula= 'Y ~ X' , data=dc).fit()
print(modeldcC2_2.summary())

# %%
