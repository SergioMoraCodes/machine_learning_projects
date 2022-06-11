import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


if __name__=='__main__':
    df_heart = pd.read_csv('Data/heart.csv')
    print(df_heart['target'].info())

    X = df_heart.drop('target', axis=1)
    y = df_heart['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # dict of classifiers
    classifiers = {
        'KNN': KNeighborsClassifier(),
        'LinearSVC': LinearSVC(),
        'SVC': SVC(),
        'SGD': SGDClassifier(),
        'DecisionTree': DecisionTreeClassifier(), # default depth=3
        'Bagging': BaggingClassifier() # default number of estimators is 10, default base_estimator is DecisionTreeClassifier()
    }

    # iterate over classifiers and print accuracy using bagging
    for name, clf in classifiers.items():
        bagging = BaggingClassifier(clf, n_estimators=5)
        bagging.fit(X_train, y_train)
        y_pred = bagging.predict(X_test)
        print('{}: {}'.format(name, accuracy_score(y_test, y_pred)))
