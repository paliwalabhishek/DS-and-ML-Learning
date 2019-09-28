import numpy as np
import pandas as pd
import seaborn as sns
sns.set_palette('husl')
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('iris.csv')
tmp = data.drop('Id',axis=1)
#g = sns.pairplot(tmp, hue='Species', markers='+')
#plt.show()

X = tmp.drop('Species', axis=1)
y = data['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=5)

k_range = list(range(1,26))
scores = []
for k in k_range:
	knn = KNeighborsClassifier(n_neighbors=k)
	knn.fit(X_train,y_train)
	y_pred = knn.predict(X_test)
	scores.append(metrics.accuracy_score(y_test,y_pred))
	#print(y_pred)

plt.plot(k_range, scores)
plt.xlabel('Value of k for KNN')
plt.ylabel('Accuracy Score')
plt.title('Accuracy Scores for Values of k of k-Nearest-Neighbors')
plt.show()

logReg = LogisticRegression()
logReg.fit(X_train,y_train)
y_pred = logReg.predict(X_test)
print(metrics.accuracy_score(y_test,y_pred))