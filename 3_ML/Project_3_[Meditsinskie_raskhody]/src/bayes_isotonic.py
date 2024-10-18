"""БЛОК ФУНКЦИЙ ЗАДАНИЯ 'Проект 3 [Медицинские расходы]'""";

"""МЕТОД ОПОРНЫХ ВЕКТОРОВ"""

from pandas import DataFrame, concat
from sklearn.isotonic import IsotonicRegression
from sklearn.naive_bayes import GaussianNB

from src.encoders import *
from src.plotter import *
from src.utils import *

"""Gaussian Naive Bayes"""

# ФУНКЦИЯ ОПРЕДЕЛЕНИЯ ЛИНЕЙНЫХ РЕГРЕССОРОВ МЕТОДОМ НАИВНОГО БАЙЕСА
def train_gnb_regressor(df_data:DataFrame, parameters:list, sc_train_y:StandardScaler, target:str="charges"):
    return GaussianNB().fit(X=sc_train_y.inverse_transform(df_data[parameters].values).astype(int),
                                      y=sc_train_y.inverse_transform(df_data[target].values.reshape(-1,1)).astype(int).ravel())



"""Isotonic Regression"""

# ФУНКЦИЯ ОПРЕДЕЛЕНИЯ Isotonic Regression
def train_iso_regressor(df_data:DataFrame, parameters:list, target:str="charges"):
    return IsotonicRegression().fit(X=df_data[parameters].values.flatten(),
                                              y=df_data[target].values)
    



# ФУНКЦИЯ ПРОГНОЗИРОВАНИЯ РЕГРЕССИИ
def iso_validation(regressors:dict, data:DataFrame, important:list, sc_train_y:StandardScaler, target:str="charges") -> DataFrame:

    # пустая таблица
    df_data = DataFrame()
    df_data.index = data.index

    df_data[target] = sc_train_y.inverse_transform(data[target].values.reshape(-1,1)).astype(int)

    # список столбцов
    df_data["predict"] = regressors.predict(data[important].values.flatten())
    df_data["predict"] = sc_train_y.inverse_transform(df_data["predict"].values.reshape(-1,1)).astype(int)
    return df_data







    
