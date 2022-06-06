import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import jaccard_score
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')


print('Libraries imported successfully')

df = pd.read_csv('../input_1500_dataset/training_dataset.csv')
print('Dataset imported successfully')


X = df["rotten_Tomatoes"].values.reshape(-1,1)
Y = df["Calculated_rating"].values.reshape(-1,1)

LogR = LogisticRegression()
LogR.fit(X,np.ravel(Y.astype(int)))

# 1) Scatter Plot 

# matplotlib scatter funcion w/ logistic regression
# plt.scatter(X,Y)
# plt.xlabel("Rotten_Tomatoes")
# plt.ylabel("Probability of Calculated Rating")
# plt.show()

# 2) Scatter with prediction value

plt.scatter(X,Y)
plt.scatter(X,LogR.predict(X))
plt.xlabel("Rotten_Tomatoes")
plt.ylabel("Probability of Calculated Rating")
plt.show()

# 3) RegPlot with dropNa
# sns.regplot(x='rotten_Tomatoes', y='Calculated_rating', data=df,dropna = True)
# plt.show()

# 4) lmPlot with hue
# sns.lmplot(x='rotten_Tomatoes', y='Calculated_rating', data=df, hue ='Status')
# plt.show()

# 5) Pairplot with Calculated rating
# sns.pairplot(df, hue ='Calculated_rating')
# plt.show()

# 6) BoxPlot - Plot by grouping status -- (pos, neg, neu)
# df.boxplot(by ='Status', column =['Calculated_rating'], grid = False)
# plt.show()