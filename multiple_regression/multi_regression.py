# Lasso : Lasso regression is a type of linear regression that uses shrinkage. 
# Shrinkage is where data values are shrunk towards a central point, like the mean. 
# The lasso procedure encourages simple, sparse models (i.e. models with fewer parameters).

import pandas as pd
import numpy as np

# for plotting
import matplotlib.pyplot as plt


# to divide train and test set
from sklearn.model_selection import train_test_split

# feature scaling
from sklearn.preprocessing import StandardScaler

# for tree binarisation
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score


# to build the models
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR



# to evaluate the models
from sklearn.metrics import mean_squared_error

pd.pandas.set_option('display.max_columns', None)

import warnings
warnings.filterwarnings('ignore')


print('Libraries imported successfully')

df = pd.read_csv('../input_1500_dataset/training_dataset.csv')
print('Dataset imported successfully')

print('Dataset shape', df.shape)

data = df[['Positive', 'Negative', 'Neutral', 'numVotes','averageRating','rotten_Tomatoes','Status','Calculated_rating']]

print('Dataframe CDF created successfully')

# print(data.isnull().mean())

df.replace([np.inf, -np.inf], np.nan, inplace=True)
# print(df.head())

# print(df.info())

# print(df.corr())

categorical = [var for var in data.columns if data[var].dtype=='O']
print(categorical)
print('There are {} categorical variables'.format(len(categorical)))

numerical = [var for var in data.columns if data[var].dtype!='O']
print('There are {} numerical variables'.format(len(numerical)))

# cdf = df[['Positive', 'Negative', 'Neutral', 'numVotes','averageRating','rotten_Tomatoes','Status']]

X_train, X_test, y_train, y_test = train_test_split(data, data.Calculated_rating, test_size=0.4, random_state=0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


X_train=pd.get_dummies(X_train,columns=categorical,drop_first=True)
X_test=pd.get_dummies(X_test,columns=categorical,drop_first=True)


# StandardScaler is used to 
# resize the distribution of values so that the mean of the
# observed values is 0 and the standard deviation is 1

scaler = StandardScaler() # create an instance
scaler.fit(X_train) #  fit  the scaler to the train set for later use
# StandardScaler(copy=True, with_mean=True, with_std=True)


# print(X_train.head())

# print(X_train.describe())


# "Ridge regression" will use all predictors in final model whereas 
# "Lasso regression" can be used for feature selection because coefficient values can be zero.

# For the Lasso random_state is due to the fitting of the regression coefficients of each variable. 
# This can be done in a 'cyclic' way, or selecting variables at random at each iteration.
# The attribute selection = 'cyclic' for the first and selection = 'random for the latter. 
# The latter involves random numbers.

print("\Lasso Model............................................\n")
# Lasso regression model
lin_model = Lasso(random_state=2909)
lin_model.fit(scaler.transform(X_train), y_train)

lin_pred_train = lin_model.predict(scaler.transform(X_train))
print('linear train mse: {}'.format(mean_squared_error(y_train, lin_pred_train)))

lin_pred_test = lin_model.predict(scaler.transform(X_test))
print('linear test mse: {}'.format(mean_squared_error(y_test, lin_pred_test)))



print("\nRidge Model............................................\n")
# Ridge Regression Model
ridgeReg = Ridge(alpha=10)

ridgeReg.fit(X_train,y_train)

#train and test scorefor ridge regression
train_score_ridge = ridgeReg.score(X_train, y_train)
test_score_ridge = ridgeReg.score(X_test, y_test)

ridge_pred_train = ridgeReg.predict(scaler.transform(X_train))
print('Ridge train mse: {}'.format(mean_squared_error(y_train, ridge_pred_train)))

ridge_pred_test = ridgeReg.predict(scaler.transform(X_test))
print('Ridge test mse: {}'.format(mean_squared_error(y_test, ridge_pred_test)))


print("The train score for ridge model is {}".format(train_score_ridge))
print("The test score for ridge model is {}".format(test_score_ridge))