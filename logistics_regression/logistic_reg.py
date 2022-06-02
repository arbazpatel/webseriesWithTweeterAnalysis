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

data = df[['Positive', 'Negative', 'Neutral', 'numVotes','averageRating','rotten_Tomatoes','Status','Calculated_rating']]
print('Dataframe CDF created successfully')

categorical = [var for var in data.columns if data[var].dtype=='O']

numerical = [var for var in data.columns if data[var].dtype!='O']

le= LabelEncoder()
y_encoded=le.fit_transform(categorical)
# print(y_encoded)

X = np.asarray(df[['Positive', 'Negative', 'Neutral', 'numVotes','averageRating','rotten_Tomatoes']])

# print(X)

y = np.asarray(data['Calculated_rating'])
y = np.asarray(y)

X = preprocessing.StandardScaler().fit(X).transform(X)

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.4, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

# Grid searching key hyperparametres for logistic regression

# define models and parameters
model = LogisticRegression()
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2']
c_values = [100, 10, 1.0, 0.1, 0.01]

# define grid search
grid = dict(solver=solvers,penalty=penalty,C=c_values)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X, y)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
yhat = LR.predict(X_test)

# print(yhat)

yhat_prob = LR.predict_proba(X_test)
# print(yhat_prob)


print('Test And Actual Result Similarity',jaccard_score(y_test, yhat,pos_label = "PAIDOFF",average='micro'))
