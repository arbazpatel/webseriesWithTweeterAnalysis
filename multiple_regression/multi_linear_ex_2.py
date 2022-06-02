import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split

print('Libraries imported successfully')

df = pd.read_csv('../input_1500_dataset/training_dataset.csv')
print('Dataset imported successfully')

print(df.shape)
data = df[['Positive', 'Negative', 'Neutral', 'numVotes','averageRating','rotten_Tomatoes','EncodedStatus','Calculated_rating']]


print('Dataframe CDF created for example 2 successfully')

print(data.isnull().mean())
df.replace([np.inf, -np.inf], np.nan, inplace=True)
# print(df.head())

# print(df.info())

print('corelation',df.corr())


x = df[['Positive', 'Negative', 'Neutral', 'numVotes','averageRating','rotten_Tomatoes']]
y = df['Calculated_rating']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 100)

mlr = LinearRegression()  
mlr.fit(x_train, y_train)

print("Intercept: ", mlr.intercept_)
print("Coefficients:", list(zip(x, mlr.coef_)))

mlr_pred_train = mlr.predict(x_train)
print('Multi-linear train mse: {}'.format(mean_squared_error(y_train, mlr_pred_train)))

mlt_pred_test = mlr.predict(x_test)
print('Multi-linear test mse: {}'.format(mean_squared_error(y_test, mlt_pred_test)))


mlt_pred_test= mlr.predict(x_test)

predd = []
import math
# for i in mlt_pred_test:
#     if i>1:
#         predd.append('Positive')
#     if i<1:
#         predd.append('Negative')
#     if i==2:
#         predd.append('Neutral')

# print("Prediction for test set: {}".format(y_pred_mlr))

# mlr_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': mlt_pred_test, 'converted':predd})

mlr_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': mlt_pred_test})

print(mlr_diff.head())