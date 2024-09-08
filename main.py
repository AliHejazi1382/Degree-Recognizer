from utils import *
import numpy as np


x_train, x_cv, x_test, y_train, y_cv, y_test = get_data()
best_alpha, best_degree = choose_best_hyperparameters(x_train, x_cv, y_train, y_cv)
print(f'{best_alpha} {best_degree}')

plot_data(x_train,y_train, 12, best_alpha)


