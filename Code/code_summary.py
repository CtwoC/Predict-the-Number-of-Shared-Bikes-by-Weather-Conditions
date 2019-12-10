#%%
# Load library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import re
import os
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from time import sleep
from selenium.common.exceptions import TimeoutException
from sklearn.tree import export_graphviz
from sklearn.model_selection import cross_val_score
from sklearn.tree import export_graphviz  
from statsmodels.formula.api import ols
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier


#%% [markdown]
# ### Scraper Part
# It is not necessary to run it, because we already save the data as csv file
#%%
# Function for get table from the website
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
# scraper weather from 365 days
bicycle17=pd.read_csv('../DataSet/bicycle17.csv')
datelist=list(bicycle17['Date'])
datelist=np.array(datelist)
datelist=np.unique(datelist)

#%%
# scraper
# Because this loop runs for a long time, we divide the year into several parts to crawl at the same time
# Slice the datelist into different parts by the slice method and run the following loops respectively

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

#%%
# combine the data by month
AugData = pd.read_csv("../DataSet/AugData.csv")
JulyData = pd.read_csv("../DataSet/JulyData.csv")
SepData = pd.read_csv("../DataSet/SepData.csv")
OctData = pd.read_csv("../DataSet/SepData.csv")
p1Data = pd.read_csv("../DataSet/weather17p1.csv")
p4Data = pd.read_csv("../DataSet/weather17p4.csv")

Weather = pd.concat([p1Data, JulyData, AugData, SepData, OctData, p4Data], axis = 0, ignore_index = True)


# %%
# Load the bicycle17 data
bicycle17 = pd.read_csv("../DataSet/bicycle17.csv")
bicycle17.columns = ['drop', 'Date', 'hour', 'cnt']
bicycle17 = bicycle17.drop(columns = ['drop'])

# %%
# Clean dataframe time function
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
# preprocess the weather dataframe
Weather.columns = ['drop', 'time', 'TF', 'Humidity', 'WindSpeed', 'weather', 'Date']
Weather['hour'] = Weather.apply(cleanDfTime, axis = 1)
Weather = Weather.drop(columns = ['drop', 'time'])
Weather = Weather.dropna()
Weather['hour'] = Weather['hour'].astype('int')
# Weather.reset_index()

#%%
# merge the weather dataframe and bicycle17 dataframe and write it in to a new csv file
Final_data = Weather.merge(bicycle17, on = ['Date','hour'], how = 'left')
Final_data['cnt'] = Final_data['cnt'].fillna(0)
Final_data['cnt'] = Final_data['cnt'].astype('int')
# Final_data.to_csv("hour2017.csv")

#%% [markdown]
# ### EDA Part

#%%
# EDA Class

class edaFunction:
    def __init__(self, dframe):
        self.dframe = dframe
        self.color = sns.color_palette()

    # check info
    def ckInfo(self):
        cnt = 1
        # info
        try:
            print(str(cnt) + ': info()')
            cnt += 1
            print(self.dframe.info())
        except:
            print("Please input a dataFrame")
        
        # describe
        print(str(cnt)+': describe(): ')
        cnt+=1
        print(self.dframe.describe())
        
        # dtype
        print(str(cnt)+': dtypes: ')
        cnt+=1
        print(self.dframe.dtypes)

        # columns
        try:
            print(str(cnt)+': columns: ')
            cnt+=1
            print(self.dframe.columns)
        except:
            print("error! No columns!")

        # head
        print(str(cnt)+': head() -- ')
        cnt+=1
        print(self.dframe.head())

        # shape
        print(str(cnt)+': shape: ')
        cnt+=1
        print(self.dframe.shape)

    # value
    def ckValues(self):
        cnt = 1
        for i in self.dframe.columns :
            print(str(cnt)+':', i, 'value_counts(): ')
            print(self.dframe[i].value_counts())
            cnt +=1

    # check NA
    def ckNA(self):
        print(self.dframe.isna().sum())
        pass

    # drop NA
    def dropNA(self):
        return self.dframe.dropna()

    # fill NA with 0
    def fillNA(self):
        return self.dframe.fillna(0)

    # plot Bar plot for season EDA
    def seasonEDA(self):
        print("The count of season bike using shows below :\n", self.dframe.groupby('season')['cnt'].sum())
        self.dframe.groupby('season')['cnt'].sum().plot(kind='bar', figsize=(6, 4), color = self.color)
        plt.ylabel('cnt')
        plt.xlabel('Season')
        plt.title('Number of shared bikes using in every Season')

    # plot Bar plot for holiday EDA
    def holidayEDA(self):
        print("The count of holiday bike using shows below :\n", self.dframe.groupby('holiday')['cnt'].mean())
        self.dframe.groupby('holiday')['cnt'].mean().plot(kind = 'bar', figsize = (6, 4), color = self.color)
        plt.ylabel('cnt')
        plt.xlabel('Holiday')
        plt.title('Average number of shared bikes using in Holiday or not')

    # plot Bar plot for workingday EDA
    def workingdayEDA(self):
        print("The count of workingday bike using shows below :\n", self.dframe.groupby('workingday')['cnt'].mean())
        self.dframe.groupby('workingday')['cnt'].mean().plot(kind = 'bar', figsize = (6, 4), color = self.color)
        plt.ylabel('cnt')
        plt.xlabel('Workingday')
        plt.title('Average number of shared bikes using in Workingday or not')

    # plot Bar plot for weather type EDA
    def weatherTypeEDA(self):
        print("The count of Weather Tpye bike using shows below :\n", self.dframe.groupby('weather')['cnt'].sum())
        self.dframe.groupby('weather')['cnt'].sum().plot(kind = 'bar', figsize = (6, 4), color = self.color)
        plt.ylabel('cnt')
        plt.xlabel('Holiday')
        plt.title('Number of shared bikes using in every Weather Type')

    # def weekendEDA(self):
    #     print("The count of weekend bike using shows below :\n", self.dframe.groupby('weekend')['cnt'].sum())
    #     self.dframe.groupby('weekend')['cnt'].sum().plot(kind = 'bar', figsize = (20, 4))

    # plot Bar plot for hour EDA
    def hourEDABar(self):
        print("The count of hour bike using shows below :\n", self.dframe.groupby('hour')['cnt'].sum())
        #ax1 = self.dframe.groupby('hour')['cnt'].sum().plot(kind = 'line', figsize = (6, 4), color = 'red', label = 'line')
        #self.dframe.groupby('hour')['cnt'].sum().plot(kind = 'bar', figsize = (6, 4), color = 'blue')
        sns.factorplot(x = "hour", y = "cnt", data = self.dframe, kind = 'bar', size = 5, aspect = 1.5)
        plt.ylabel('cnt')
        plt.xlabel('Hour')
        plt.title('Number of shared bikes using in every Hour')
    
    # plot line plot for hour EDA
    def hourEDAline(self):
        sns.factorplot(x = "hour", y = "cnt", data = self.dframe, kind = 'point', size = 5, aspect = 1.5)
        plt.ylabel('cnt')
        plt.xlabel('Hour')
        plt.title('Number of shared bikes using in every Hour')

    # plot Box plot for user EDA
    def registerBox(self):
        # fig,axes = plt.subplots(2,1)
        registered = list(self.dframe.registered)
        casual = list(self.dframe.casual)
        plt.boxplot([registered,casual],positions=[1,2])
        plt.ylabel('cnt')
        plt.title('Boxplot of Number of Casual User and Registered User')
        
    # plot Violin plot for user EDA
    def registerVio(self):
        registered = list(self.dframe.registered)
        casual = list(self.dframe.casual)
        plt.violinplot([registered,casual],positions=[1,2])
        plt.ylabel('cnt')
        plt.title('Violinplot of Number of Casual User and Registered User')
    
    # plot Bar plot for day EDA
    def dayEDA(self):
        print("The count of day bike using shows below :\n", self.dframe.groupby('day')['cnt'].sum())
        # self.dframe.groupby('day')['cnt'].sum().plot(kind = 'bar', figsize = (6, 4), color = 'blue')
        sns.factorplot(x = "day", y = "cnt", data = self.dframe, kind = 'bar', size = 5, aspect = 1.5)
        plt.ylabel('cnt')
        plt.xlabel('Day')
        plt.title('Number of shared bikes using in every weekDay')
    
    # plot Bar plot for month EDA
    def monthEDA(self):
        print("The count of month bike using shows below :\n", self.dframe.groupby('month')['cnt'].sum())
        # self.dframe.groupby('month')['cnt'].sum().plot(kind = 'bar', figsize = (6, 4), color = 'blue')
        sns.factorplot(x = "month", y = "cnt", data = self.dframe, kind = 'bar', size = 5, aspect = 1.5)
        plt.ylabel('cnt')
        plt.xlabel('Month')
        plt.title('Number of shared bikes using in every Month')

    # plot Bar plot for year EDA
    def yearEDA(self):
        print("The count of year bike using shows below :\n", self.dframe.groupby('year')['cnt'].sum())
        self.dframe.groupby('year')['cnt'].sum().plot(kind = 'bar', figsize = (6, 4), color = self.color)
        plt.ylabel('cnt')
        plt.xlabel('Year')
        plt.title('Number of shared bikes using in every Year')

    # plot Bar plot for condition EDA
    def hists(self):
        fig,axes = plt.subplots(2,2)
        axes[0,0].hist(x = "TF", data = self.dframe, edgecolor = "black", linewidth = 2)
        axes[0,0].set_title("Variation of Temperature")
        axes[0,1].hist(x = "TFF", data = self.dframe, edgecolor = "black",linewidth = 2)
        axes[0,1].set_title("Variation of Temperature Feels")
        axes[1,0].hist(x = "WindSpeed", data = self.dframe, edgecolor = "black",linewidth = 2)
        axes[1,0].set_title("Variation of WindSpeed")
        axes[1,1].hist(x = "Humidity", data = self.dframe, edgecolor = "black", linewidth = 2)
        axes[1,1].set_title("Variation of Humidity")
        fig.set_size_inches(10,10)
    
    # plot Bar plot for condition EDA (data 2017)
    def hists2(self):
        fig,axes = plt.subplots(2,2)
        axes[0,0].hist(x = "TF", data = self.dframe, edgecolor = "black", linewidth = 2)
        axes[0,0].set_title("Variation of Temperature")
        axes[1,0].hist(x = "WindSpeed", data = self.dframe, edgecolor = "black",linewidth = 2)
        axes[1,0].set_title("Variation of WindSpeed")
        axes[1,1].hist(x = "Humidity", data = self.dframe, edgecolor = "black", linewidth = 2)
        axes[1,1].set_title("Variation of Humidity")
        fig.set_size_inches(10,10)
    
    # plot correlation matrix for EDA
    def correlationMatrix(self):
        cor_mat= self.dframe[:].corr()
        mask = np.array(cor_mat)
        mask[np.tril_indices_from(mask)] = False
        fig=plt.gcf()
        fig.set_size_inches(30,12)
        sns.heatmap(data = cor_mat, mask = mask, square = True, annot = True, cbar = True)

#%% 
# Load file for eda and model
dc = pd.read_csv("../DataSet/hour.csv")
dc2017 = pd.read_csv("../DataSet/hour2017.csv")

#%%
# Preprocess the dataframe
dc = dc.rename(columns={'yr':'year', 'mnth':'month', 'hr':'hour', 'weekday':'day', 'weathersit':'weather', 'temp':'TF', 'atemp':'TFF', 'hum':'Humidity', 'windspeed':'WindSpeed'})

dc2017['TF'] = dc2017['TF'].map(lambda x: x.rstrip('F'))
dc2017['Humidity'] = dc2017['TF'].map(lambda x: x.rstrip('%'))
dc2017['WindSpeed'] = dc2017['TF'].map(lambda x: x.rstrip('mph'))
dc2017['TF'] = dc2017['TF'].astype('float')
dc2017['Humidity'] = dc2017['TF'].astype('float')
dc2017['WindSpeed'] = dc2017['TF'].astype('float')

# Normalize new data (Same as normalization method used for old data)
maxTF = max(dc2017.TF)
maxHM = max(dc2017.Humidity)
maxWS = max(dc2017.WindSpeed)
dc2017['TF'] = dc2017['TF'].map(lambda x: x / maxTF)
dc2017['Humidity'] = dc2017['Humidity'].map(lambda x: x / maxHM)
dc2017['WindSpeed'] = dc2017['WindSpeed'].map(lambda x: x / maxWS)

# Instantiate eda object
dcEDA = edaFunction(dc)
dc2017EDA = edaFunction(dc2017)

#%%
# check Info
dcEDA.ckInfo()
dc2017EDA.ckInfo()
#%%
# check NA
dcEDA.ckNA()
dc2017EDA.ckNA()
#%%
# Drop NA
dcEDA.dropNA()
dc2017EDA.dropNA()
#%%
# Season EDA of dc
dcEDA.seasonEDA()

#%%
# Holiday EDA of dc
dcEDA.holidayEDA()

#%%
# Workingday EDA of dc
dcEDA.workingdayEDA()

#%%
# Weather EDA of dc
dcEDA.weatherTypeEDA()

#%%
# Hour EDA of dc
dcEDA.hourEDABar()

#%%
dcEDA.hourEDAline()

#%%
# Hour EDA of dc2017
dc2017EDA.hourEDABar()

#%%
dc2017EDA.hourEDAline()

#%%
# Day EDA of dc
dcEDA.dayEDA()

#%%
# Month EDA of dc
dcEDA.monthEDA()

#%%
# Year EDA of dc
dcEDA.yearEDA()

#%%
# Condition EDA of dc
dcEDA.hists()

#%%
# Condition EDA of dc2017
dc2017EDA.hists2()

#%%
# User EDA of dc
dcEDA.registerBox()

#%%
dcEDA.registerVio()

#%%
# correlationMatrix for dc
dcEDA.correlationMatrix()

#%%
# correlationMatrix for dc2017
dc2017EDA.correlationMatrix()

#%% [markdown]
# ### Model Part

#%%
# Create dummy variables for categorial type variable 'weather' and 'season'

Wtype_dummies = pd.get_dummies(dc['weather'], prefix='w')
dc= pd.concat([dc, Wtype_dummies], axis=1)
Season_dummies = pd.get_dummies(dc['season'], prefix='s')
dc= pd.concat([dc, Season_dummies], axis=1)


#%%
# regression tree
# Part A: (Y~CNT, X~ALL DAYS)
# Prepare our X data (features, predictors, regressors) and y data (target, dependent variable)

xdc_A = dc[['s_1', 's_2', 's_3', 'hour','holiday','WindSpeed', 'workingday', 'TF', 'Humidity']]
ydc_A = dc['cnt']

# Split dataset into 80% train, 20% test
X_train, X_test, y_train, y_test= train_test_split(xdc_A, ydc_A, test_size=0.2, random_state=1)

#%%
# seaborn plot
sns.set()
sns.pairplot(xdc_A)

#%%
# Instantiate a DecisionTreeRegressor 'regtree0'
regtree1 = DecisionTreeRegressor(max_depth=4, min_samples_leaf=0.05,random_state=22) # set minimum leaf to contain at least 10% of data points

# Fit regtree0 to the training set
regtree1.fit(X_train, y_train)  

# evaluation
y_pred1 = regtree1.predict(X_test)  # Compute y_pred
mse_regtree1 = MSE(y_test, y_pred1)  # Compute mse_regtree1
rmse_regtree1 = mse_regtree1 ** (.5) # Compute rmse_regtree1
regtree1score1 = regtree1.score(X_test, y_test)

print("Test set score of regtree1: {:.2f}".format(regtree1score1))
print("Test set RMSE of regtree1: {:.2f}".format(rmse_regtree1))


# %%
# Let us compare the performance with OLS

olsdc1 = linear_model.LinearRegression() 
olsdc1.fit( X_train, y_train )

y_pred_ols1 = olsdc1.predict(X_test)  # Predict test set labels/values
olsdc1score1 = olsdc1.score(X_test, y_test)

mse_ols1 = MSE(y_test, y_pred_ols1)  # Compute mse_ols
rmse_ols1 = mse_ols1**(0.5)  # Compute rmse_ols

print('Linear Regression test set score1: {:.2f}'.format(olsdc1score1))
print('Linear Regression test set RMSE1: {:.2f}'.format(rmse_ols1))
print('Regression Tree test set score1: {:.2f}'.format(regtree1score1))
print('Regression Tree test set RMSE1: {:.2f}'.format(rmse_regtree1))

#
#%%
# Instantiate a DecisionTreeRegressor dt
SEED = 28
regtree1 = DecisionTreeRegressor(max_depth=10, min_samples_leaf=0.022, random_state=SEED)

# Evaluate the list of MSE ontained by 10-fold CV
# Set n_jobs to -1 in order to exploit all CPU cores in computation
MSE_CV1 = - cross_val_score(regtree1, X_train, y_train, cv= 10, scoring='neg_mean_squared_error', n_jobs = -1)
regtree1.fit(X_train, y_train)  # Fit 'regtree1' to the training set
y_predict_train = regtree1.predict(X_train)  # Predict the labels of training set
y_predict_test = regtree1.predict(X_test)  # Predict the labels of test set

print('CV RMSE1:', MSE_CV1.mean()**(0.5) )  #CV MSE 
print('Training set RMSE1:', MSE(y_train, y_predict_train)**(0.5) )   # Training set MSE
print('Test set RMSE1:', MSE(y_test, y_predict_test)**(0.5) )   # Test set MSE 

#%%
print('Training set RMSE1:', MSE(y_train, y_predict_train)**(0.5) )   # Training set MSE
print('Test set RMSE1:', MSE(y_test, y_predict_test)**(0.5) )   # Test set MSE print('CV RMSE:', MSE_CV.mean()**(0.5) )  #CV MSE 
print('Training set RMSE1:', MSE(y_train, y_predict_train)**(0.5) )   # Training set MSE
print('Test set RMSE1:', MSE(y_test, y_predict_test)**(0.5) )   # Test set MSE 

#%%
# Graphing the tree
filepath = os.path.join('tree1')
export_graphviz(regtree1, out_file = filepath+'.dot' , feature_names =['s_1', 's_2', 's_3', 'hour','holiday','WindSpeed', 'workingday', 'TF', 'Humidity']) 

#%%
# regression tree
# Part B: (Y~CNT, X~only workingday)
# Prepare our X data (features, predictors, regressors) and y data (target, dependent variable)

xdc_B = dc[['s_1', 's_2', 's_3', 'hour','WindSpeed', 'workingday', 'TF', 'Humidity']]
ydc_B = dc['cnt']

# Split dataset into 80% train, 20% test
X_train, X_test, y_train, y_test= train_test_split(xdc_B, ydc_B, test_size=0.2, random_state=1)

#%%
# seaborn plot
sns.set()
sns.pairplot(xdc_B)

#%%
# Instantiate a DecisionTreeRegressor 'regtree0'
regtree2 = DecisionTreeRegressor(max_depth=4, min_samples_leaf=0.05,random_state=22) # set minimum leaf to contain at least 10% of data points

# Fit regtree0 to the training set
regtree2.fit(X_train, y_train)  

# evaluation
y_pred2 = regtree2.predict(X_test)  # Compute y_pred
mse_regtree2 = MSE(y_test, y_pred2)  # Compute mse_regtree2
rmse_regtree2 = mse_regtree2 ** (.5) # Compute rmse_regtree2
print("Test set RMSE of regtree2: {:.2f}".format(rmse_regtree2))


# %%
# Let us compare the performance with OLS

olsdc2 = linear_model.LinearRegression() 
olsdc2.fit( X_train, y_train )

olsdc2score2 = olsdc2.score(X_test, y_test)
regtree2score2 = regtree2.score(X_test, y_test)

y_pred_ols2 = olsdc2.predict(X_test)  # Predict test set labels/values

mse_ols2 = MSE(y_test, y_pred_ols2)  # Compute mse_ols
rmse_ols2 = mse_ols2**(0.5)  # Compute rmse_ols

print('Linear Regression test set RMSE2: {:.2f}'.format(rmse_ols2))
print('Regression Tree test set RMSE2: {:.2f}'.format(rmse_regtree2))
print('Linear Regression test set score2: {:.2f}'.format(olsdc2score2))
print('Regression Tree test set score2: {:.2f}'.format(regtree2score2))
#
#%%
# Instantiate a DecisionTreeRegressor dt
SEED = 28
regtree2 = DecisionTreeRegressor(max_depth=10, min_samples_leaf=0.022, random_state=SEED)

# Evaluate the list of MSE ontained by 10-fold CV
# Set n_jobs to -1 in order to exploit all CPU cores in computation
MSE_CV2 = - cross_val_score(regtree2, X_train, y_train, cv= 10, scoring='neg_mean_squared_error', n_jobs = -1)
regtree2.fit(X_train, y_train)  # Fit 'regtree1' to the training set
y_predict_train = regtree2.predict(X_train)  # Predict the labels of training set
y_predict_test = regtree2.predict(X_test)  # Predict the labels of test set

print('CV RMSE2:', MSE_CV2.mean()**(0.5) )  #CV MSE 
print('Training set RMSE2:', MSE(y_train, y_predict_train)**(0.5) )   # Training set MSE
print('Test set RMSE2:', MSE(y_test, y_predict_test)**(0.5) )   # Test set MSE 

#%%
print('Training set RMSE2:', MSE(y_train, y_predict_train)**(0.5) )   # Training set MSE
print('Test set RMSE2:', MSE(y_test, y_predict_test)**(0.5) )   # Test set MSE print('CV RMSE:', MSE_CV.mean()**(0.5) )  #CV MSE 
print('Training set RMSE2:', MSE(y_train, y_predict_train)**(0.5) )   # Training set MSE
print('Test set RMSE2:', MSE(y_test, y_predict_test)**(0.5) )   # Test set MSE 

#%%
# Graphing the tree
filepath = os.path.join('tree2')
export_graphviz(regtree2, out_file = filepath+'.dot' , feature_names =['s_1', 's_2', 's_3', 'hour','WindSpeed', 'workingday', 'TF', 'Humidity']) 

#%%
# regression tree
# Part C: (Y~casual, X~ALL DAYS)
# Prepare our X data (features, predictors, regressors) and y data (target, dependent variable)

xdc_C = dc[['s_1', 's_2', 's_3', 'hour','holiday','WindSpeed', 'workingday', 'TF', 'Humidity']]
ydc_C = dc['casual']

# Split dataset into 80% train, 20% test
X_train, X_test, y_train, y_test= train_test_split(xdc_C, ydc_C, test_size=0.2, random_state=1)

#%%
# seaborn plot
sns.set()
sns.pairplot(xdc_C)

#%%
# Instantiate a DecisionTreeRegressor 'regtree0'
regtree3 = DecisionTreeRegressor(max_depth=4, min_samples_leaf=0.05,random_state=22) # set minimum leaf to contain at least 10% of data points

# Fit regtree0 to the training set
regtree3.fit(X_train, y_train)  

# evaluation
y_pred3 = regtree3.predict(X_test)  # Compute y_pred
mse_regtree3 = MSE(y_test, y_pred3)  # Compute mse_regtree0
rmse_regtree3 = mse_regtree3 ** (.5) # Compute rmse_regtree0
print("Test set RMSE of regtree3: {:.2f}".format(rmse_regtree3))


# %%
# Let us compare the performance with OLS

olsdc3 = linear_model.LinearRegression() 
olsdc3.fit( X_train, y_train )

olsdc3score3 = olsdc3.score(X_test, y_test)
regtree3score3 = regtree3.score(X_test, y_test)

y_pred_ols3 = olsdc3.predict(X_test)  # Predict test set labels/values

mse_ols3 = MSE(y_test, y_pred_ols3)  # Compute mse_ols
rmse_ols3 = mse_ols3**(0.5)  # Compute rmse_ols

print('Linear Regression test set RMSE3: {:.2f}'.format(rmse_ols3))
print('Regression Tree test set RMSE3: {:.2f}'.format(rmse_regtree3))
print('Linear Regression test set score3: {:.2f}'.format(olsdc3score3))
print('Regression Tree test set score3: {:.2f}'.format(regtree3score3))
#
#%%
# Instantiate a DecisionTreeRegressor dt
SEED = 28
regtree3 = DecisionTreeRegressor(max_depth=10, min_samples_leaf=0.022, random_state=SEED)

# Evaluate the list of MSE ontained by 10-fold CV
# Set n_jobs to -1 in order to exploit all CPU cores in computation
MSE_CV3 = - cross_val_score(regtree3, X_train, y_train, cv= 10, scoring='neg_mean_squared_error', n_jobs = -1)
regtree3.fit(X_train, y_train)  # Fit 'regtree1' to the training set
y_predict_train = regtree3.predict(X_train)  # Predict the labels of training set
y_predict_test = regtree3.predict(X_test)  # Predict the labels of test set

print('CV RMSE3:', MSE_CV3.mean()**(0.5) )  #CV MSE 
print('Training set RMSE3:', MSE(y_train, y_predict_train)**(0.5) )   # Training set MSE
print('Test set RMSE3:', MSE(y_test, y_predict_test)**(0.5) )   # Test set MSE 

#%%
print('Training set RMSE3:', MSE(y_train, y_predict_train)**(0.5) )   # Training set MSE
print('Test set RMSE3:', MSE(y_test, y_predict_test)**(0.5) )   # Test set MSE print('CV RMSE:', MSE_CV.mean()**(0.5) )  #CV MSE 
print('Training set RMSE3:', MSE(y_train, y_predict_train)**(0.5) )   # Training set MSE
print('Test set RMSE3:', MSE(y_test, y_predict_test)**(0.5) )   # Test set MSE 

#%%
# Graphing the tree
filepath = os.path.join('tree3')
export_graphviz(regtree3, out_file = filepath+'.dot' , feature_names =['s_1', 's_2', 's_3', 'hour','holiday','WindSpeed', 'workingday', 'TF', 'Humidity']) 

#%%
# regression tree
# Part D: (Y~casual, X~only workingday)
# Prepare our X data (features, predictors, regressors) and y data (target, dependent variable)

xdc_D = dc[['s_1', 's_2', 's_3', 'hour','WindSpeed', 'workingday', 'TF', 'Humidity']]
ydc_D = dc['casual']

# Split dataset into 80% train, 20% test
X_train, X_test, y_train, y_test= train_test_split(xdc_D, ydc_D, test_size=0.2, random_state=1)

#%%
# seaborn plot
sns.set()
sns.pairplot(xdc_D)

#%%
# Instantiate a DecisionTreeRegressor 'regtree0'
regtree4 = DecisionTreeRegressor(max_depth=4, min_samples_leaf=0.05,random_state=22) # set minimum leaf to contain at least 10% of data points

# Fit regtree0 to the training set
regtree4.fit(X_train, y_train)  

# evaluation
y_pred4 = regtree4.predict(X_test)  # Compute y_pred
mse_regtree4 = MSE(y_test, y_pred4)  # Compute mse_regtree0
rmse_regtree4 = mse_regtree4 ** (.5) # Compute rmse_regtree0
print("Test set RMSE4 of regtree4: {:.2f}".format(rmse_regtree4))


# %%
# Let us compare the performance with OLS

olsdc4 = linear_model.LinearRegression() 
olsdc4.fit( X_train, y_train )

olsdc4score4 = olsdc4.score(X_test, y_test)
regtree4score4 = regtree4.score(X_test, y_test)

y_pred_ols4 = olsdc4.predict(X_test)  # Predict test set labels/values

mse_ols4 = MSE(y_test, y_pred_ols4)  # Compute mse_ols
rmse_ols4 = mse_ols4**(0.5)  # Compute rmse_ols

print('Linear Regression test set RMSE4: {:.2f}'.format(rmse_ols4))
print('Regression Tree test set RMSE4: {:.2f}'.format(rmse_regtree4))
print('Linear Regression test set score4: {:.2f}'.format(olsdc4score4))
print('Regression Tree test set score4: {:.2f}'.format(regtree4score4))

#
#%%
# Instantiate a DecisionTreeRegressor dt
SEED = 28
regtree4 = DecisionTreeRegressor(max_depth=10, min_samples_leaf=0.022, random_state=SEED)

# Evaluate the list of MSE ontained by 10-fold CV
# Set n_jobs to -1 in order to exploit all CPU cores in computation
MSE_CV4 = - cross_val_score(regtree4, X_train, y_train, cv= 10, scoring='neg_mean_squared_error', n_jobs = -1)
regtree4.fit(X_train, y_train)  # Fit 'regtree1' to the training set
y_predict_train = regtree4.predict(X_train)  # Predict the labels of training set
y_predict_test = regtree4.predict(X_test)  # Predict the labels of test set

print('CV RMSE4:', MSE_CV4.mean()**(0.5) )  #CV MSE 
print('Training set RMSE4:', MSE(y_train, y_predict_train)**(0.5) )   # Training set MSE
print('Test set RMSE4:', MSE(y_test, y_predict_test)**(0.5) )   # Test set MSE 

#%%
print('Training set RMSE4:', MSE(y_train, y_predict_train)**(0.5) )   # Training set MSE
print('Test set RMSE4:', MSE(y_test, y_predict_test)**(0.5) )   # Test set MSE print('CV RMSE:', MSE_CV.mean()**(0.5) )  #CV MSE 
print('Training set RMSE4:', MSE(y_train, y_predict_train)**(0.5) )   # Training set MSE
print('Test set RMSE4:', MSE(y_test, y_predict_test)**(0.5) )   # Test set MSE 

#%%
# Graphing the tree

filepath = os.path.join('tree4')
export_graphviz(regtree4, out_file = filepath+'.dot' , feature_names =['s_1', 's_2', 's_3', 'hour','WindSpeed', 'workingday', 'TF', 'Humidity']) 

#%%
# regression tree
# Part E: (Y~CNT, X~ALL DAYS) (Train ~ DC, test ~ 2017data)
# Prepare our X data (features, predictors, regressors) and y data (target, dependent variable)

# Using DC set as train, 2017 data as test
X_train = dc[['TF', 'Humidity', 'WindSpeed']]
y_train = dc['cnt']

X_test = dc2017[['TF', 'Humidity', 'WindSpeed']]
y_test = dc2017['cnt']


#%%
# Instantiate a DecisionTreeRegressor 'regtree'
regtree5 = DecisionTreeRegressor(max_depth=4, min_samples_leaf=0.05,random_state=22) # set minimum leaf to contain at least 10% of data points

# Fit regtree0 to the training set
regtree5.fit(X_train, y_train)  

# evaluation
y_pred5 = regtree5.predict(X_test)  # Compute y_pred
mse_regtree5 = MSE(y_test, y_pred5)  # Compute mse_regtree1
rmse_regtree5 = mse_regtree5 ** (.5) # Compute rmse_regtree1
print("Test set RMSE of regtree5: {:.2f}".format(rmse_regtree5))


# %%
# Let us compare the performance with OLS

olsdc5 = linear_model.LinearRegression() 
olsdc5.fit( X_train, y_train )

olsdc5score5 = olsdc5.score(X_test, y_test)
regtree5score5 = regtree5.score(X_test, y_test)

y_pred_ols5 = olsdc5.predict(X_test)  # Predict test set labels/values

mse_ols5 = MSE(y_test, y_pred_ols5)  # Compute mse_ols
rmse_ols5 = mse_ols5**(0.5)  # Compute rmse_ols

print('Linear Regression test set RMSE5: {:.2f}'.format(rmse_ols5))
print('Regression Tree test set RMSE5: {:.2f}'.format(rmse_regtree5))
print('Linear Regression test set score5: {:.2f}'.format(olsdc5score5))
print('Regression Tree test set score5: {:.2f}'.format(regtree5score5))
#
#%%
# Instantiate a DecisionTreeRegressor dt
SEED = 28
regtree5 = DecisionTreeRegressor(max_depth=10, min_samples_leaf=0.022, random_state=SEED)

# Evaluate the list of MSE ontained by 10-fold CV
# Set n_jobs to -1 in order to exploit all CPU cores in computation
MSE_CV5 = - cross_val_score(regtree5, X_train, y_train, cv= 10, scoring='neg_mean_squared_error', n_jobs = -1)
regtree5.fit(X_train, y_train)  # Fit 'regtree1' to the training set
y_predict_train = regtree5.predict(X_train)  # Predict the labels of training set
y_predict_test = regtree5.predict(X_test)  # Predict the labels of test set

print('CV RMSE5:', MSE_CV1.mean()**(0.5) )  #CV MSE 
print('Training set RMSE5:', MSE(y_train, y_predict_train)**(0.5) )   # Training set MSE
print('Test set RMSE5:', MSE(y_test, y_predict_test)**(0.5) )   # Test set MSE 

#%%
print('Training set RMSE5:', MSE(y_train, y_predict_train)**(0.5) )   # Training set MSE
print('Test set RMSE5:', MSE(y_test, y_predict_test)**(0.5) )   # Test set MSE print('CV RMSE:', MSE_CV.mean()**(0.5) )  #CV MSE 
print('Training set RMSE5:', MSE(y_train, y_predict_train)**(0.5) )   # Training set MSE
print('Test set RMSE5:', MSE(y_test, y_predict_test)**(0.5) )   # Test set MSE 

#%%
# Graphing the tree

filepath = os.path.join('tree5')
export_graphviz(regtree5, out_file = filepath+'.dot' , feature_names =['TF', 'Humidity', 'WindSpeed']) 


# %%

