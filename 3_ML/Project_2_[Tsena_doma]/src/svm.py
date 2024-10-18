"""БЛОК ФУНКЦИЙ ЗАДАНИЯ 'Проект 2 [Цена дома]'""";

"""МЕТОД ОПОРНЫХ ВЕКТОРОВ"""

from pandas import DataFrame
from sklearn.svm import SVR

from src.encoders import *
from src.plotter import *
from src.utils import *


# ФУНКЦИЯ ОБУЧЕНИЯ МЕТОДА ОПОРНЫХ ВЕКТОРОВ
def train_svm_regressor(data:DataFrame, parameters:list, target:str="SalePrice",
                         kernel:str="linear", degree:int=1, epsilon:float=0.1) -> dict:

    regression_dict = {}

    parameters = parameters[1:] if target in parameters else parameters

    clf = SVR(kernel=kernel,  degree=degree,  epsilon=epsilon)

    regression_dict["regressor"]  = clf.fit(X=data[parameters].values, y=data[target].values)
    regression_dict["parameters"] = data[parameters].columns

    return regression_dict


# ФУНКЦИЯ ФОРМИРОВАНИЯ ТАБЛИЦЫ
def svm_validation(regressor:dict, data:DataFrame, target:str="SalePrice") -> DataFrame:

    # пустая таблица
    df_data = DataFrame()
    df_data["Id"] = data["Id"]
    df_data[target] = data[target]

    # построение прогноза
    # заполнение результата
    df_data["predict"] = regressor["regressor"].predict(data[regressor["parameters"]].values)\
                                                  .reshape(-1,1).astype(int)
    
    # таблица результатов
    return df_data

