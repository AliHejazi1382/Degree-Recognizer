import numpy as np

import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt
dlc = dict(dlblue = '#0096ff', dlorange = '#FF9300', dldarkred='#C00000', dlmagenta='#FF40FF', dlpurple='#7030A0', dldarkblue =  '#0D5BDC')

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
    for degree in range(1, 11):
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        x_train_mapped = poly.fit_transform(x_train)

        scalar = StandardScaler()
        x_train_mapped_scaled = scalar.fit_transform(x_train_mapped)

        model = LinearRegression()
        model.fit(x_train_mapped_scaled, y_train)

        yhat = model.predict(x_train_mapped_scaled)
        train_mses.append(mean_squared_error(y_train, yhat) / 2)

        x_cv_mapped = poly.transform(x_cv)
        x_cv_mapped_scaled = scalar.transform(x_cv_mapped)

        yhat = model.predict(x_cv_mapped_scaled)
        cv_mses.append(mean_squared_error(y_cv, yhat) / 2)

    return train_mses, cv_mses

def choose_best_hyperparameters(x_train, x_cv, y_train, y_cv):
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(include_bias=False)),
        ('scaler', StandardScaler()),
        ('model', Ridge())
    ])
    
    # Updated parameter grid with larger alpha values to avoid small values that cause instability
    param_grid = {
        'poly__degree': range(1, 11),
        'model__alpha': [1e-3, 1e-2, 0.1, 1, 10, 100]
    }
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
    
    grid_search.fit(x_train, y_train)
    
    best_degree = grid_search.best_params_['poly__degree']
    best_lambda = grid_search.best_params_['model__alpha']
    
    return best_lambda, best_degree

def predict(x_train, y_train, degree, alpha):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    x_train_mapped = poly.fit_transform(x_train)

    scaler = StandardScaler()
    x_train_mapped_scaled = scaler.fit_transform(x_train_mapped)

    model = Ridge(alpha=alpha)
    model.fit(x_train_mapped_scaled, y_train)
    return model.predict(x_train_mapped_scaled)

def plot_data(x_train, y_train, best_degree, best_alpha):
    # Sort the data
    sorted_indices = np.argsort(x_train.flatten())
    x_train_sorted = x_train[sorted_indices]
    y_train_sorted = y_train[sorted_indices]

    yhat = predict(x_train_sorted, y_train_sorted, best_degree, best_alpha)

    plt.figure(figsize=(8, 6))
    plt.scatter(x_train_sorted, y_train_sorted, color='blue', label='Training Data')
    plt.plot(x_train_sorted, yhat, color='red', label=f'Best Fit: degree={best_degree}, alpha={best_alpha}')
    plt.title('Polynomial Regression Fit')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    
    plt.show()