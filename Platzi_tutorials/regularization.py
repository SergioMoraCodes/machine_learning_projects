import pandas as pd
import sklearn

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ =='__main__':

    df = pd.read_csv('./Data/felicidad.csv')

    X = df[['gdp','family','lifexp','freedom', 'corruption','generosity','dystopia']]
    y = df[['score']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    model_linear     = LinearRegression().fit(X_train, y_train)
    y_predict_linear = model_linear.predict(X_test)

    model_lasso      = Lasso(alpha=0.02).fit(X_train, y_train)
    y_predict_lasso  = model_lasso.predict(X_test)

    model_ridge      = Ridge(alpha=1).fit(X_train, y_train)
    y_predict_ridge  = model_ridge.predict(X_test)


    linear_loss = mean_squared_error(y_test, y_predict_linear)
    print('linear loss: ', linear_loss)

    lasso_loss  = mean_squared_error(y_test, y_predict_lasso)
    print('lasso loss: ', lasso_loss)

    ridge_loss  = mean_squared_error(y_test, y_predict_ridge)
    print('ridge loss: ', ridge_loss)

    print('='*32)
    print('COEF lasso: ',model_lasso.coef_)

    print('='*32)
    print('COEF ridge: ',model_ridge.coef_)