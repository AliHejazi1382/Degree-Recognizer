from utils import *
import numpy as np

x_train, x_cv, x_test, y_train, y_cv, y_test = get_data()

train_mses, cv_mses =  choose_degree(x_train, x_cv, y_train, y_cv)
plot_Jcv_Jtrain(train_mses, cv_mses)
print(cv_mses)
print(train_mses)