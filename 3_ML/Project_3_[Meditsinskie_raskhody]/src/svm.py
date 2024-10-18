"""БЛОК ФУНКЦИЙ ЗАДАНИЯ 'Проект 3 [Медицинские расходы]'""";

"""МЕТОД ОПОРНЫХ ВЕКТОРОВ"""

from pandas import DataFrame
from sklearn.svm import SVR

from src.encoders import *
from src.plotter import *
from src.utils import *


# ФУНКЦИЯ ОБУЧЕНИЯ МЕТОДА ОПОРНЫХ ВЕКТОРОВ
def train_svm_regressor(data:DataFrame, parameters:list, target:str="charges",
                         kernel:str="linear", degree:int=1, epsilon:float=0.1) -> dict:

    regression_dict = {}

    clf = SVR(kernel=kernel,  degree=degree,  epsilon=epsilon)

    regression_dict["regressor"]  = clf.fit(X=data[parameters].values, y=data[target].values)
    regression_dict["parameters"] = data[parameters].columns

    return regression_dict


# ФУНКЦИЯ ФОРМИРОВАНИЯ ТАБЛИЦЫ
def svm_validation(regressor:dict, data:DataFrame, sc_train_y:StandardScaler, target:str="charges") -> DataFrame:

    # пустая таблица
    df_data = DataFrame()
    # df_data["Id"] = data["Id"]
    df_data.index = data.index
    df_data[target] = sc_train_y.inverse_transform(data[target].values.reshape(-1,1))

    # построение прогноза
    # заполнение результата
    df_data["predict"] = regressor["regressor"].predict(data[regressor["parameters"]].values)\
                                                  .reshape(-1,1)
    
    df_data["predict"] =  sc_train_y.inverse_transform(df_data["predict"].values.reshape(-1,1)).astype(int)
    
    # таблица результатов
    return df_data

