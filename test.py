import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score,mean_squared_error

crop_dataset = pd.read_csv('Dataset/Dataset.csv')
crop_dataset.fillna(0, inplace = True)
crop_dataset['Production'] = crop_dataset['Production'].astype(np.int64)

le = LabelEncoder()
crop_dataset['State_Name'] = pd.Series(le.fit_transform(crop_dataset['State_Name']))
crop_dataset['District_Name'] = pd.Series(le.fit_transform(crop_dataset['District_Name']))
crop_dataset['Season'] = pd.Series(le.fit_transform(crop_dataset['Season']))
crop_dataset['Crop'] = pd.Series(le.fit_transform(crop_dataset['Crop']))
crop_datasets = crop_dataset.values
cols = crop_datasets.shape[1]-1
X = crop_datasets[:,0:cols]
Y = crop_datasets[:,cols]
Y = Y.astype('uint8')

X = normalize(X)

#X = X.reshape(-1, 1)
#Y = Y.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

print(X)
print(Y)
clf = DecisionTreeRegressor(max_depth=100,random_state=0,max_leaf_nodes=20,max_features=5,splitter="random")
clf.fit(X,Y)
predict = clf.predict(X_test)
print(predict)
#score = clf.score(predict,Y)
#print(score)

test = pd.read_csv('Dataset/test.csv')
test.fillna(0, inplace = True)
test['State_Name'] = pd.Series(le.fit_transform(test['State_Name']))
test['District_Name'] = pd.Series(le.fit_transform(test['District_Name']))
test['Season'] = pd.Series(le.fit_transform(test['Season']))
test['Crop'] = pd.Series(le.fit_transform(test['Crop']))
test = test.values
test = normalize(test)
cols = test.shape[1]
test = test[:,0:cols]
print(clf.predict(test))

mse = mean_squared_error(predict,y_test)
rmse = np.sqrt(mse);
print(rmse)


