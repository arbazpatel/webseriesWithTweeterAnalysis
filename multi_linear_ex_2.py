import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split

print('Libraries imported successfully')

df = pd.read_csv('./training_dataset_1000.csv')
print('Dataset imported successfully')

print(df.shape)
data = df[['Positive', 'Negative', 'Neutral', 'numVotes','averageRating','rotten_Tomatoes','EncodedStatus']]


print('Dataframe CDF created for example 2 successfully')

print(data.isnull().mean())
df.replace([np.inf, -np.inf], np.nan, inplace=True)
# print(df.head())

# print(df.info())

print('corelation',df.corr())


x = df[['Positive', 'Negative', 'Neutral', 'numVotes','averageRating','rotten_Tomatoes']]
y = df['EncodedStatus']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 100)

mlr = LinearRegression()  
mlr.fit(x_train, y_train)

print("Intercept: ", mlr.intercept_)
print("Coefficients:", list(zip(x, mlr.coef_)))


y_pred_mlr= mlr.predict(x_test)

predd = []
import math
for i in y_pred_mlr:
    if i>1:
        predd.append('Positive')
    if i<1:
        predd.append('Negative')
    if i==2:
        predd.append('Neutral')

# print("Prediction for test set: {}".format(y_pred_mlr))

mlr_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred_mlr, 'converted':predd})

# mlr_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': predd})

print(mlr_diff.head())