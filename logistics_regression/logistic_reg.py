import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


print('Libraries imported successfully')

df = pd.read_csv('../input_1500_dataset/training_dataset.csv')
print('Dataset imported successfully')

data = df[['Positive', 'Negative', 'Neutral', 'numVotes','averageRating','rotten_Tomatoes','Status']]
print('Dataframe CDF created successfully')

categorical = [var for var in data.columns if data[var].dtype=='O']

numerical = [var for var in data.columns if data[var].dtype!='O']

from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
y_encoded=le.fit_transform(categorical)
# print(y_encoded)


X = np.asarray(df[['Positive', 'Negative', 'Neutral', 'numVotes','averageRating','rotten_Tomatoes']])

# print(X)

y = np.asarray(data['Status'])
y = np.asarray(y)


from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
yhat = LR.predict(X_test)

# print(yhat)

yhat_prob = LR.predict_proba(X_test)
# print(yhat_prob)

from sklearn.metrics import jaccard_similarity_score

print('Test And Actual Result Similarity',jaccard_similarity_score(y_test, yhat))