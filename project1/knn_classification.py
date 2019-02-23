import pandas as pd
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score

data = pd.read_csv("training.csv")

X = data.iloc[:, 1:]
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


random_grid = {
    'n_neighbors':[int(x) for x in np.linspace(start = 5, stop = 31, num = 20)],
    'metric': ['manhattan','minkowski','euclidean'],
    'weights': ['uniform','distance'],
    'algorithm': ['auto','ball_tree', 'kd_tree', 'brute'],
    'leaf_size': [int(x) for x in np.linspace(start=30, stop=200, num = 10)]
               }
pprint(random_grid)
knc = KNeighborsClassifier()
knc_random = RandomizedSearchCV(estimator=knc, 
	param_distributions=random_grid, n_iter=100, cv=3,
	verbose=2, random_state=42, n_jobs=-1)
knc_random.fit(X_train, y_train)
print(knc_random.best_params_)
best_random = knc_random.best_estimator_
y_predict_knc = knc_random.predict(X_test)
y_predict_knc_best_random = best_random.predict(X_test)

acc_random = accuracy_score(y_test, y_predict_knc_best_random)
acc_knc = accuracy_score(y_test, y_predict_knc)
print(acc_knc)


#print(rfc.get_params(deep=True))
