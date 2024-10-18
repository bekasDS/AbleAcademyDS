"""БЛОК ФУНКЦИЙ ЗАДАНИЯ 'Проект 2 [Цена дома]'""";

"""СЛУЧАЙНЫЙ ЛЕС"""

from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from src.encoders import *
from src.plotter import *
from src.utils import *


# ФУНКЦИЯ ОБУЧЕНИЯ МОДЕЛИ 'СЛУЧАЙНЫЙ ЛЕС'
def train_random_forest_regressor(data:DataFrame, parameters:list, target:str="SalePrice", n_estimators:int=1, random_state:int=0):
    
    tree_regressor = {}
    parameters = parameters[1:] if target in parameters else parameters

    regressor = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    tree_regressor["regressor"]  = regressor.fit(X=data[parameters].values, y=data[target].values)
    tree_regressor["parameters"] = data[parameters].columns

    return tree_regressor



# ФУНКЦИЯ ПРЕДСКАЗАНИЯ 'SalePrice' ПО СЛУЧАЙНОМУ ЛЕСУ
def random_forest_validation(model:dict, data:DataFrame, target:str="SalePrice"):
    # пустая таблица
    df_data = DataFrame()
    df_data["Id"] = data["Id"]
    df_data[target] = data[target]

    # построение прогноза
    # заполнение результата
    df_data["predict"] = model["regressor"].predict(data[model["parameters"]].values)       # данные

    # таблица результатов
    return df_data




# *****************************
"""ДЕРЕВО ПРИНЯТИЯ РЕШЕНИЙ"""

# ФУНКЦИЯ ОБУЧЕНИЯ МОДЕЛИ РЕГРЕССИИ ДЕРЕВО ПРИНЯТИЯ РЕШЕНИЙ
def train_decision_tree_regressor(data:DataFrame, parameters:list, target:str="SalePrice",
                                  degree:int=1, random_state:int=0) -> dict:

    tree_regressor = {}
    parameters = parameters[1:] if target in parameters else parameters

    regressor = DecisionTreeRegressor(random_state=random_state)
    tree_regressor["regressor"]  = regressor.fit(X=data[parameters].values, y=data[target].values)
    tree_regressor["parameters"] = data[parameters].columns

    return tree_regressor


