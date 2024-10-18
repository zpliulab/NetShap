import copy

import numpy as np
import pandas as pd
import sklearn
import inspect
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import norm
import os


# Additional functions

class Data:
    def __init__(self, data, col_names):
        self.data = data
        # self.indx = data.index
        self.col_names = col_names
        self.n = data.shape[0]
        self.weights = np.ones(self.n)
        self.weights /= self.n


def convert_to_data(value):
    if isinstance(value, Data):
        return value
    elif type(value) == np.ndarray:
        return Data(value, [str(i) for i in range(value.shape[1])])
    elif str(type(value)).endswith("pandas.core.series.Series'>"):
        return Data(value.values.reshape((1, len(value))), value.index.tolist())
    elif str(type(value)).endswith("pandas.core.frame.DataFrame'>"):
        return Data(value.values, value.columns.tolist())
    else:
        assert False, str(type(value)) + "is currently not a supported format type"

    # Convert model to standard model class


class Model:
    def __init__(self, f):
        self.f = f


def convert_to_model(value):
    if isinstance(value, Model):
        return value
    else:
        return Model(value)


class Link:
    def __init__(self):
        pass


class IdentityLink(Link):
    def __str__(self):
        return "identity"

    @staticmethod
    def f(x):
        return x

    @staticmethod
    def finv(x):
        return x


class LogitLink(Link):
    def __str__(self):
        return "logit"

    @staticmethod
    def f(x):
        return np.log(x / (1 - x))

    @staticmethod
    def finv(x):
        return 1 / (1 + np.exp(-x))


def convert_to_link(val):
    if isinstance(val, Link):
        return val
    elif val == "identity":
        return IdentityLink()
    elif val == "logit":
        return LogitLink()
    else:
        assert False, "Passed link object must be a subclass of iml.Link"



def absolute_value(shapley_score, y):
    """计算dataset的shapley平均值"""
    label = copy.deepcopy(y)  # 使用深拷贝以避免修改原始数据
    label[label == 0] = -1
    ave_shapley = np.round(np.sum(np.abs(shapley_score.values * label.reshape(-1, 1)), axis=0)
                           /shapley_score.shape[0], 4).reshape(1, -1)
    ave_shapley = pd.DataFrame(ave_shapley, columns=shapley_score.columns)
    return ave_shapley

def remove_the_least_value(neighbors, shapley_value, y):
    """去掉集合中shapley排序最末的元素"""
    absolute_shapley = np.array(absolute_value(shapley_value, y))[0]
    minshap = np.min(absolute_shapley)
    least_index = np.where(absolute_shapley == minshap)[0]
    if len(least_index) == len(neighbors):
        neighbors = neighbors[:-1]
        return neighbors
    neighbors = [n for j, n in enumerate(neighbors) if j not in least_index]
    return neighbors



def summary_plot(shap_value, features = None, feature_names = None, sort=True, plot_type='dot', show=False):
    import shap
    shap_value = np.array(shap_value)
    shap.summary_plot(shap_value, features,
                      feature_names=feature_names, sort=sort, plot_type=plot_type, show=show)
    if plot_type=='dot':
        plt.xlabel("local Shapley value")
        plt.show()
    else:
        plt.xlabel("gloable Shapley value")
        plt.show()


def sampling(X, nsamples=50, random_state=5):
    if nsamples >= X.shape[0]:
        return X
    else:
        return sklearn.utils.resample(X, n_samples=nsamples, random_state=random_state)


def save_shap_value(output_dir, shapley_value, feature_name, name:str):
    # print(X_data.shape)
    shapley_value.to_csv(path_or_buf=output_dir + '/' + name + 'shapvalue.csv')
    nodes_file = output_dir + '/' + 'selected_nodes.csv'
    mid = pd.DataFrame(feature_name)
    print(nodes_file)
    mid.to_csv(path_or_buf=nodes_file, header=False, index=False)


def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]








