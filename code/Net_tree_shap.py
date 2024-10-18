import ctypes
import numpy as np
import glob
from shap.explainers import Tree
import pandas as pd
import sys


def Ntreeshap(model, X, background, start):
    ensemble = Tree(model, data=background).model

    if ensemble.values.shape[2] > 1:
        values = np.ascontiguousarray(ensemble.values[..., 1])
    else:
        values = np.ascontiguousarray(ensemble.values[..., 0])
    if type(X) == pd.DataFrame:
        X = np.ascontiguousarray(X)
    if type(background) == pd.DataFrame:
        background = np.ascontiguousarray(background)

    Nx = X.shape[0]
    Nz = background.shape[0]
    Nt = ensemble.features.shape[0]
    d = X.shape[1]
    depth = ensemble.features.shape[1]

    results = np.zeros(X.shape)

    libfile = glob.glob('./code/NetTreeshap.so')

    # Open the shared library
    mylib = ctypes.CDLL(libfile[0])
    mylib.main_treeshap.restype = ctypes.c_int
    mylib.main_treeshap.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_int,
                                    np.ctypeslib.ndpointer(dtype=np.float64),
                                    np.ctypeslib.ndpointer(dtype=np.float64),
                                    np.ctypeslib.ndpointer(dtype=np.float64),
                                    np.ctypeslib.ndpointer(dtype=np.float64),
                                    np.ctypeslib.ndpointer(dtype=np.int32),
                                    np.ctypeslib.ndpointer(dtype=np.int32),
                                    np.ctypeslib.ndpointer(dtype=np.int32),
                                    np.ctypeslib.ndpointer(dtype=np.float64), ctypes.c_int]

    mylib.main_treeshap(Nx, Nz, Nt, d, depth, X, background,
                        ensemble.thresholds, values,
                        ensemble.features, ensemble.children_left, ensemble.children_right, results, len(start))

    results = results[:, len(start):]
    return results
