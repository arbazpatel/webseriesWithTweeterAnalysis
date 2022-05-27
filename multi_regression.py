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
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR



# to evaluate the models
from sklearn.metrics import mean_squared_error

pd.pandas.set_option('display.max_columns', None)

import warnings
warnings.filterwarnings('ignore')


print('Libraries imported successfully')

df = pd.read_csv('./training_dataset_1000.csv')
print('Dataset imported successfully')

print(df.shape)
data = df[['Positive', 'Negative', 'Neutral', 'numVotes','averageRating','rotten_Tomatoes','Status']]
print('Dataframe CDF created successfully')

print(data.isnull().mean())
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

X_train, X_test, y_train, y_test = train_test_split(data, data.averageRating, test_size=0.2, random_state=0)
print(X_train.shape, X_test.shape, X_train.shape, X_test.shape)


X_train=pd.get_dummies(X_train,columns=categorical,drop_first=True)
X_test=pd.get_dummies(X_test,columns=categorical,drop_first=True)

scaler = StandardScaler() # create an instance
scaler.fit(X_train) #  fit  the scaler to the train set for later use
# StandardScaler(copy=True, with_mean=True, with_std=True)


print(X_train.head())

print(X_train.describe())

lin_model = Lasso(random_state=2909)
lin_model.fit(scaler.transform(X_train), y_train)

pred = lin_model.predict(scaler.transform(X_train))
print('linear train mse: {}'.format(mean_squared_error(y_train, pred)))
pred = lin_model.predict(scaler.transform(X_test))
print('linear test mse: {}'.format(mean_squared_error(y_test, pred)))


importance = pd.Series(np.abs(lin_model.coef_.ravel()))
importance.index = X_train.columns
importance.sort_values(inplace=True, ascending=False)
importance.plot.bar(figsize=(18,6))