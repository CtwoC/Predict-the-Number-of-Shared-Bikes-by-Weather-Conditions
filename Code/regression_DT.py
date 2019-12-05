#%%
import pandas as pd
import os
from statsmodels.formula.api import ols

from sklearn.tree import DecisionTreeRegressor  # Import DecisionTreeRegressor
from sklearn.model_selection import train_test_split  # Import train_test_split
from sklearn.metrics import mean_squared_error as MSE  # Import mean_squared_error as MSE
from sklearn import linear_model

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
# Part A:
# Prepare our X data (features, predictors, regressors) and y data (target, dependent variable)

xdc = dc[['s_1', 's_2', 's_3', 'holiday', 'workingday', 'TF', 'Humidity']]
ydc = dc['cnt']

# Split dataset into 80% train, 20% test
X_train, X_test, y_train, y_test= train_test_split(xdc, ydc, test_size=0.2, random_state=1)

#%%
# seaborn plot
import seaborn as sns
sns.set()
sns.pairplot(xdc)

#%%
# Instantiate a DecisionTreeRegressor 'regtree0'
regtree0 = DecisionTreeRegressor(max_depth=4, min_samples_leaf=0.05,random_state=22) # set minimum leaf to contain at least 10% of data points

# Fit regtree0 to the training set
regtree0.fit(X_train, y_train)  

# evaluation
y_pred = regtree0.predict(X_test)  # Compute y_pred
mse_regtree0 = MSE(y_test, y_pred)  # Compute mse_regtree0
rmse_regtree0 = mse_regtree0 ** (.5) # Compute rmse_regtree0
print("Test set RMSE of regtree0: {:.2f}".format(rmse_regtree0))


# %%
# Let us compare the performance with OLS

olspizza = linear_model.LinearRegression() 
olspizza.fit( X_train, y_train )

y_pred_ols = olspizza.predict(X_test)  # Predict test set labels/values

mse_ols = MSE(y_test, y_pred_ols)  # Compute mse_ols
rmse_ols = mse_ols**(0.5)  # Compute rmse_ols

print('Linear Regression test set RMSE: {:.2f}'.format(rmse_ols))
print('Regression Tree test set RMSE: {:.2f}'.format(rmse_regtree0))

#
#%%
# Instantiate a DecisionTreeRegressor dt
SEED = 28
regtree1 = DecisionTreeRegressor(max_depth=10, min_samples_leaf=0.022, random_state=SEED)

# Evaluate the list of MSE ontained by 10-fold CV
from sklearn.model_selection import cross_val_score
# Set n_jobs to -1 in order to exploit all CPU cores in computation
MSE_CV = - cross_val_score(regtree1, X_train, y_train, cv= 10, scoring='neg_mean_squared_error', n_jobs = -1)
regtree1.fit(X_train, y_train)  # Fit 'regtree1' to the training set
y_predict_train = regtree1.predict(X_train)  # Predict the labels of training set
y_predict_test = regtree1.predict(X_test)  # Predict the labels of test set

print('CV RMSE:', MSE_CV.mean()**(0.5) )  #CV MSE 
print('Training set RMSE:', MSE(y_train, y_predict_train)**(0.5) )   # Training set MSE
print('Test set RMSE:', MSE(y_test, y_predict_test)**(0.5) )   # Test set MSE 

#%%
print('Training set RMSE:', MSE(y_train, y_predict_train)**(0.5) )   # Training set MSE
print('Test set RMSE:', MSE(y_test, y_predict_test)**(0.5) )   # Test set MSE print('CV RMSE:', MSE_CV.mean()**(0.5) )  #CV MSE 
print('Training set RMSE:', MSE(y_train, y_predict_train)**(0.5) )   # Training set MSE
print('Test set RMSE:', MSE(y_test, y_predict_test)**(0.5) )   # Test set MSE 

#%%
# Graphing the tree
from sklearn.tree import export_graphviz  

filepath = os.path.join('tree1')
export_graphviz(regtree1, out_file = filepath+'.dot' , feature_names =['s_1', 's_2', 's_3', 'holiday', 'workingday', 'TF', 'Humidity']) 

# %%
