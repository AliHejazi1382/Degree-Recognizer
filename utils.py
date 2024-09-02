import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def get_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "data/data_w3_ex1.csv")
    
    data = np.loadtxt(data_path, delimiter=',')
    x = np.expand_dims(data[:, 0], axis=1)
    y = np.expand_dims(data[:, 1], axis=1)

    x_train, x_, y_train, y_ = train_test_split(x, y, test_size=0.4, random_state=1)
    x_cv, x_test, y_cv, y_test = train_test_split(x_, y_, test_size=0.5, random_state=1)
    del x_, y_


    return x_train, x_cv, x_test, y_train, y_cv, y_test


def choose_degree(x_train, x_cv, y_train, y_cv):
    train_mses = []
    cv_mses = []
    models = []
    polys = []
    scalers = []
    for degree in range(1, 11):
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        x_train_mapped = poly.fit_transform(x_train)
        polys.append(poly)

        scalar = StandardScaler()
        x_train_mapped_scaled = scalar.fit_transform(x_train_mapped)
        scalers.append(scalar)

        model = LinearRegression()
        model.fit(x_train_mapped_scaled, y_train)
        models.append(model)

        yhat = model.predict(x_train_mapped_scaled)
        train_mses.append(mean_squared_error(y_train, yhat) / 2)

        x_cv_mapped = poly.transform(x_cv)
        x_cv_mapped_scaled = scalar.transform(x_cv_mapped)

        yhat = model.predict(x_cv_mapped_scaled)
        cv_mses.append(mean_squared_error(y_cv, yhat) / 2)
