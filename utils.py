import numpy as np
import os
from sklearn.model_selection import train_test_split

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

get_data()