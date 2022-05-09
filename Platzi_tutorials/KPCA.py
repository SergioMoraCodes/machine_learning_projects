import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt

from sklearn.decomposition import KernelPCA # !!
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler # normalizar los datos
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    df_heart = pd.read_csv('.\Data\heart.csv')
    print(df_heart.head(3))

    # the data most be saparated between features & target
    df_features = df_heart.drop(['target'], axis = 1)
    df_target   = df_heart['target']

    # Para realizar el PCA se deben normalizar los datos
    # en este caso utilizo StandardScaler
    df_features = StandardScaler().fit_transform(df_features) #load data, fit the model and applies the transformation

    # separate the data between Train, Test and validation
    X_train, X_test, y_train, y_test = train_test_split(df_features, df_target, test_size=0.3, random_state=42)

    kpca = KernelPCA(n_components=4, kernel='poly') # linear, poly, rbf(gaussian)
    kpca.fit(X_train)

    df_train = kpca.transform(X_train)
    df_test  = kpca.transform(X_test)

    logistic = LogisticRegression(solver="lbfgs")

    logistic.fit(df_train, y_train)
    print("score = ", logistic.score(df_test, y_test))