#%%
import pandas as pd
import os
from statsmodels.formula.api import ols

from sklearn.tree import DecisionTreeRegressor  # Import DecisionTreeRegressor
from sklearn.model_selection import train_test_split  # Import train_test_split
from sklearn.metrics import mean_squared_error as MSE  # Import mean_squared_error as MSE
from sklearn import linear_model

# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

#%%
# load file
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
import seaborn as sns
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
from sklearn.model_selection import cross_val_score
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
from sklearn.tree import export_graphviz  

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
import seaborn as sns
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
from sklearn.model_selection import cross_val_score
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
from sklearn.tree import export_graphviz  

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
import seaborn as sns
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
from sklearn.model_selection import cross_val_score
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
from sklearn.tree import export_graphviz  

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
import seaborn as sns
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
from sklearn.model_selection import cross_val_score
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
from sklearn.tree import export_graphviz  

filepath = os.path.join('tree4')
export_graphviz(regtree4, out_file = filepath+'.dot' , feature_names =['s_1', 's_2', 's_3', 'hour','WindSpeed', 'workingday', 'TF', 'Humidity']) 

#%%
# regression tree
# Part E: (Y~CNT, X~ALL DAYS) (Train ~ DC, test ~ 2017data)
# Prepare our X data (features, predictors, regressors) and y data (target, dependent variable)

# load file
#dc2017 = pd.read_csv("hour2017.csv")
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
from sklearn.model_selection import cross_val_score
# Set n_jobs to -1 in order to exploit all CPU cores in computation
MSE_CV5 = - cross_val_score(regtree5, X_train, y_train, cv= 10, scoring='neg_mean_squared_error', n_jobs = -1)
regtree5.fit(X_train, y_train)  # Fit 'regtree1' to the training set
y_predict_train = regtree5.predict(X_train)  # Predict the labels of training set
y_predict_test = regtree5.predict(X_test)  # Predict the labels of test set

print('CV RMSE5:', MSE_CV5.mean()**(0.5) )  #CV MSE 
print('Training set RMSE5:', MSE(y_train, y_predict_train)**(0.5) )   # Training set MSE
print('Test set RMSE5:', MSE(y_test, y_predict_test)**(0.5) )   # Test set MSE 

#%%
print('Training set RMSE5:', MSE(y_train, y_predict_train)**(0.5) )   # Training set MSE
print('Test set RMSE5:', MSE(y_test, y_predict_test)**(0.5) )   # Test set MSE print('CV RMSE:', MSE_CV.mean()**(0.5) )  #CV MSE 
print('Training set RMSE5:', MSE(y_train, y_predict_train)**(0.5) )   # Training set MSE
print('Test set RMSE5:', MSE(y_test, y_predict_test)**(0.5) )   # Test set MSE 

#%%
# Graphing the tree
from sklearn.tree import export_graphviz  

filepath = os.path.join('tree5')
export_graphviz(regtree5, out_file = filepath+'.dot' , feature_names =['TF', 'Humidity', 'WindSpeed']) 


# %%
