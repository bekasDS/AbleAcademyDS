"""БЛОК ФУНКЦИЙ ЗАДАНИЯ 'Проект 2 [Цена дома]'""";

"""МЕТОД ОПОРНЫХ ВЕКТОРОВ"""

from pandas import DataFrame, concat
from sklearn.isotonic import IsotonicRegression
from sklearn.naive_bayes import GaussianNB

from src.encoders import *
from src.plotter import *
from src.utils import *

"""Gaussian Naive Bayes"""

# ФУНКЦИЯ ОПРЕДЕЛЕНИЯ ЛИНЕЙНЫХ РЕГРЕССОРОВ МЕТОДОМ НАИВНОГО БАЙЕСА
def train_gnb_regressor(df_data:DataFrame, parameters:list, target:str="SalePrice"):

    parameters = parameters[1:] if target in parameters else parameters

    return GaussianNB().fit(X=df_data[parameters].values, y=df_data[target].values)



"""Isotonic Regression"""

# ФУНКЦИЯ ОПРЕДЕЛЕНИЯ Isotonic Regression
def train_iso_regressor(df_data:DataFrame, parameters:list, target:str="SalePrice"):
    parameters = parameters[1:] if target in parameters else parameters

    # print(df_data[parameters].values.shape)

    # print(df_data[target].values.shape)

    return IsotonicRegression().fit(X=df_data[parameters].values.flatten(),
                                              y=df_data[target].values)
    



# ФУНКЦИЯ ПРОГНОЗИРОВАНИЯ РЕГРЕССИИ
def iso_validation(regressors:dict, data:DataFrame, important:list, target:str="SalePrice") -> DataFrame:

    # пустая таблица
    df_data = DataFrame()

    # список столбцов
    df_data["predict"] = regressors.predict(data[important].values.flatten()).astype(int)

    # # объединение столбцов
    return concat([data["Id"], data[target], df_data], axis=1)







    
