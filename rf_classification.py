import pandas as pd
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score

data = pd.read_csv("training.csv")


X = data.iloc[:, 1:]
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['log2', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [2, 5, 10]
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)

rfc = RandomForestClassifier()
rfc_random = RandomizedSearchCV(estimator = rfc, 
    param_distributions = random_grid, n_iter = 100, cv = 3,
    verbose=2, random_state=42, n_jobs = -1)
rfc_random.fit(X_train, y_train)
print(rfc_random.best_params_)
best_random = rfc_random.best_estimator_
y_predict_rfc = rfc_random.predict(X_test)
y_predict_rfc_best_random = best_random.predict(X_test)

acc_random = accuracy_score(y_test, y_predict_rfc_best_random)
acc_rfc = accuracy_score(y_test, y_predict_rfc)
print(acc_rfc)


#print(rfc.get_params(deep=True))
