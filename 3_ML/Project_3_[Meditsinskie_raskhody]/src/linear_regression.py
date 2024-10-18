"""БЛОК ФУНКЦИЙ ЗАДАНИЯ 'Проект 3 [Медицинские расходы]'""";

"""ЛИНЕЙНАЯ РЕГРЕССИЯ"""

from pandas import DataFrame, concat
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from src.encoders import *
from src.plotter import *
from src.utils import *


# ФУНКЦИЯ ОПРЕДЕЛЕНИЯ ЛИНЕЙНЫХ РЕГРЕССОРОВ
def train_linear_regressor(df_data:DataFrame, important:list|str, target:str="charges"):
    # объект линейной регрессии
    regressor = LinearRegression()

    if len(important) == 1:
        return regressor.fit(X=df_data[important].values.reshape(-1,1), y=df_data[target].values)
    else:
        return regressor.fit(X=df_data[important].values, y=df_data[target].values)



# ФУНКЦИЯ ОПРЕДЕЛЕНИЯ ЛИНЕЙНЫХ РЕГРЕССОРОВ
def train_simple_lin_regressors(df_data:DataFrame, important:list, target:str="charges") -> dict:
    
    regressors_dict = {}

    important.remove(target)
    
    for column in important:
        regressors_dict[column] = train_linear_regressor(df_data=df_data, important=[column])
    return regressors_dict



# ФУНКЦИЯ ПРОГНОЗИРОВАНИЯ РЕГРЕССИИ
def linear_validation(regressors:dict, data:DataFrame, sc_train_y:StandardScaler,
                            #  linear_encoder_dict:dict,
                              target:str="charges",
                             isSimple=True,  # тип регрессии 
                             isScale=True,    # обратная трансформация
                             **kwargs) -> DataFrame:

    df_data = data.copy()
    # целевой столбец
    df_data[target] = sc_train_y.inverse_transform(data[target].values.reshape(-1,1)).ravel()

    # обратная трансформация исходных данных
    # по столбцам
    if isSimple:
        # заполнение значениями
        for key in regressors:
            if key != target:
                # предсказание
                df_data[key] = data[key].apply(lambda x: regressors[key].predict([[x]])[0].reshape(-1,1).ravel()[0])
                
                # обратная трансформация исходных данных
                if isScale:
                    df_data[key] =  sc_train_y.inverse_transform(df_data[key].values.reshape(-1,1))
                

    # все столбцы
    else:
        # список столбцов
        important = kwargs.get('important', None)
        # print(important)
        df_data["predict"] = regressors.predict(data[important].values)
        if isScale:
            df_data["predict"] = sc_train_y.inverse_transform(df_data["predict"].values.reshape(-1,1))
        # df_data = df_data.drop(columns=important)
        df_data = df_data[[target, "predict"]].astype(int)

    return df_data




# *********************************
"""Polynomial Linear Regression"""

# ФУНКЦИЯ ОБУЧЕНИЯ МОДЕЛИ МНОЖЕСТВЕННОЙ ЛИНЕЙНОЙ ПОЛИНОМИАЛЬНОЙ РЕГРЕССИИ
def train_multiple_lin_poly_regressor(data:DataFrame, parameters:list, target:str="charges",
                                      degree:int=1, interaction_only:bool=False) -> dict:
    
    parameters = parameters[1:] if target in parameters else parameters

    poly_regressors_dict = {}
    poly_regressors_dict["degree"] = degree


    feature_generator = PolynomialFeatures(degree=degree, interaction_only=interaction_only)
    poly_regressors_dict["feature"] = feature_generator
    
    poly_regressors_dict["parameters"] = data[parameters].columns


    poly_feature = feature_generator.fit_transform(data[parameters].values)
    
    poly_regressors = LinearRegression()
    poly_regressors_dict["regressor"] = poly_regressors.fit(X=poly_feature,
                                                            y=data[target].values)

    return poly_regressors_dict



# ФУНКЦИЯ ПРЕДСКАЗАНИЯ 'charges' ПО МНОЖЕСТВЕННОЙ ПОЛИНОМИАЛЬНОЙ ЛИНЕЙНОЙ РЕГРЕССИИ
def multiple_polynomial_linear_validation(model:dict, data:DataFrame, sc_train_y:StandardScaler, target:str="charges"):
    # пустая таблица
    df_data = DataFrame()
    # df_data["Id"] = data["Id"]
    df_data.index = data.index
    df_data[target] = sc_train_y.inverse_transform(data[target].values.reshape(-1,1))
    
    # название колонки
    # colnames = "-".join(model["parameters"])
    colnames = "predict"

    # построение прогноза
    feature_generator = model["feature"]
    poly_feature = feature_generator.fit_transform(X=data[model["parameters"]].values)

    # заполнение результата
    df_data[colnames] = model["regressor"].predict(poly_feature)       # данные
    df_data[colnames] = df_data[colnames].astype(int)
    df_data[colnames] = sc_train_y.inverse_transform(df_data[colnames].values.reshape(-1,1)).astype(int)

    # таблица результатов
    return df_data


