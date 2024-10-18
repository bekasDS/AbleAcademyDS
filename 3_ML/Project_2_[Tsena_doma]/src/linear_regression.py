"""БЛОК ФУНКЦИЙ ЗАДАНИЯ 'Проект 2 [Цена дома]'""";

"""ЛИНЕЙНАЯ РЕГРЕССИЯ"""

from pandas import DataFrame, concat
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from src.encoders import *
from src.plotter import *
from src.utils import *


# ФУНКЦИЯ ОПРЕДЕЛЕНИЯ ЛИНЕЙНЫХ РЕГРЕССОРОВ
def train_linear_regressor(df_data:DataFrame, important:list|str, target:str="SalePrice"):
    # объект линейной регрессии
    regressor = LinearRegression()

    if len(important) == 1:
        return regressor.fit(X=df_data[important].values.reshape(-1,1), y=df_data[target].values)
    else:
        return regressor.fit(X=df_data[important].values, y=df_data[target].values)



# # ФУНКЦИЯ ОПРЕДЕЛЕНИЯ ЛИНЕЙНЫХ РЕГРЕССОРОВ
# def train_linear_regressor(df_data:DataFrame, important:list|str, target:str="SalePrice", isStandard:bool=True):
#     # объект линейной регрессии
#     regressor = LinearRegression()

#     if len(important) == 1:
#         X = df_data[important].values.reshape(-1,1)
#     else:
#         X = df_data[important].values

#     y = df_data[target].values


#     if isStandard:
#         return create_scaled_linear_regression_model(X_train=X, y_train=y)

#         return regressor.fit(X=X, y=y)




# ФУНКЦИЯ ОПРЕДЕЛЕНИЯ ЛИНЕЙНЫХ РЕГРЕССОРОВ
def train_simple_lin_regressors(df_data:DataFrame, important:list, target:str="SalePrice") -> dict:
    
    regressors_dict = {}
    
    for column in important:
        regressors_dict[column] = train_linear_regressor(df_data=df_data, important=[column])
    return regressors_dict



# ФУНКЦИЯ ПРОГНОЗИРОВАНИЯ РЕГРЕССИИ
def linear_validation(regressors:dict, data:DataFrame, target:str="SalePrice", isSimple=True, **kwargs) -> DataFrame:

    # пустая таблица
    df_data = DataFrame()

    # по столбцам
    if isSimple:
        # заполнение значениями
        for key in regressors:
            df_data[key] =  data[key].apply(lambda x: int(regressors[key].predict([[x]])[0]) )
    # все столбцы
    else:
        # список столбцов
        important = kwargs.get('important', None)
        # print(important)
        df_data["predict"] = regressors.predict(data[important].values).astype(int)

    # # объединение столбцов
    return concat([data["Id"], data[target], df_data], axis=1)





# *********************************
"""Polynomial Linear Regression"""

# ФУНКЦИЯ ОБУЧЕНИЯ МОДЕЛИ МНОЖЕСТВЕННОЙ ЛИНЕЙНОЙ ПОЛИНОМИАЛЬНОЙ РЕГРЕССИИ
def train_multiple_lin_poly_regressor(data:DataFrame, parameters:list, target:str="SalePrice",
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



# ФУНКЦИЯ ПРЕДСКАЗАНИЯ 'SalePrice' ПО МНОЖЕСТВЕННОЙ ПОЛИНОМИАЛЬНОЙ ЛИНЕЙНОЙ РЕГРЕССИИ
def multiple_polynomial_linear_validation(model:dict, data:DataFrame, target:str="SalePrice"):
    # пустая таблица
    df_data = DataFrame()
    df_data["Id"] = data["Id"]
    df_data[target] = data[target]
    
    # название колонки
    # colnames = "-".join(model["parameters"])
    colnames = "predict"

    # построение прогноза
    feature_generator = model["feature"]
    poly_feature = feature_generator.fit_transform(X=data[model["parameters"]].values)

    # заполнение результата
    df_data[colnames] = model["regressor"].predict(poly_feature)       # данные
    df_data[colnames] = df_data[colnames].astype(int)

    # таблица результатов
    return df_data


