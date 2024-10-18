
"""БЛОК ФУНКЦИЙ Проекта 4 [Стоимость мобильного телефона]"""

from pandas import read_csv, options, DataFrame, Series, concat, get_dummies
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# ФУНКЦИЯ ЗАГРУЗКИ ДАННЫХ
def load_data(file:str) -> DataFrame:
    return read_csv(file)


# ФУНКЦИЯ ВСТАВКИ КОЛОНКИ НУМЕРАЦИИ
def set_numbercolumn(df:DataFrame, columnname:str="№", loc:int=0) -> DataFrame:
    df.insert(loc=loc, column=columnname, value=[i+1 for i in range(len(df.index))])
    return df.set_index(columnname)


# ФУНКЦИЯ МАСШТАБИРОВАНИЕ ПО ГРУППАМ
def scale_data(df_data:DataFrame,             # текучая для обработки
                        df_train:DataFrame,            # тренировочная таблица
                        groups_int_dict:dict,           # словарь с данными на разбиение на группы
                        target:str="price_range",
                        isTrain:bool=True,              # назначение выборки 'train/test'
                        step_dict:dict={},              # словарь с шагом разбивки на группы
                        columns_list:list=[]
                    ) -> DataFrame:
    df_encoded = df_data.copy()




    # функция распределения категорий
    def get_groups(x:float, key:str, numgroups:int,
                   step_dict:dict,
                   groups_int_dict:dict,
                   columns_list:list,
                   df_series:Series):
        max_value = df_series.max()

        # отбор по назначению выборки
        if isTrain:
            # шаг разбивки на группы
            step = round(len(df_series.unique()) / numgroups)

            # словарь с разбивкой по шагам для деления на группы
            step_dict[key] = [step, max_value]
        else:
            step = step_dict[key][0]
            max_value = step_dict[key][1]

        # распределение
        for i in np.arange(0, ceil(max_value), step):
            
            # print(i, type(i), x, type(x))  # отладка

            # формирование меток по условию
            if x <= i:
                resx = "less" + str(i)
                break
            else:
                resx = str(i) + "_" + str(i + step)
        else:
            resx = "more" + str(ceil(max_value))
        
        return resx
    

    # print(df_encoded.columns)

    # заполнение закодированными данными
    for key in groups_int_dict:

        # шифрование данных по столбцу
        df_encoded[key] = df_encoded[key].astype(object).\
                                          apply(lambda x: get_groups(x=x,
                                                                     key=key,
                                                                     numgroups=groups_int_dict[key],
                                                                     groups_int_dict=groups_int_dict,
                                                                     step_dict=step_dict,
                                                                     columns_list=columns_list,
                                                                     df_series=df_train[key]))
        # проверка закодированных данных
    # df_encoded


    # кодирование столбцов
    dummies_list = []
    for col in groups_int_dict:
        # Применение метода get_dummies() к категориальному столбцу
        df_dummies = get_dummies(df_encoded[col], prefix=f"{col}", dtype=int)

        # print(df_dummies)

        dummies_list.append(df_dummies)
        df_encoded.drop(col, axis=1, inplace=True)

    # добавление в начало списка таблицы
    dummies_list.insert(0, df_encoded)
        

    # сборка датасет
    df_FINAL = concat(dummies_list, axis=1)
    
    # проверка одинаковости столбцов
    if isTrain == False:
        missing_columns = np.setdiff1d(columns_list, df_FINAL.columns.to_list())

        # заполнение отсутствующих данных нулями
        if len(missing_columns) > 0:
            df_FINAL[missing_columns] = 0

    # смещение таргет в начало таблицы
    df_FINAL.insert(0, target, df_FINAL.pop(target))

    return step_dict, df_FINAL



# ФУНКЦИЯ масштабирования
def minmax_encoder(df_data:DataFrame, target:str="price_range") -> list:
    # стандартизация с трансформацией массива

    df_transformed = df_data.copy()

    columns_list = df_transformed.columns.to_list()

    # columns_list = df_data.columns.to_list()
    columns_list.remove(target)

    # объект масштабирования
    # 1. все столбцы
    sc_x = MinMaxScaler()
    X = df_transformed[columns_list].values
    df_transformed[columns_list] = sc_x.fit_transform(X=X)

    # 2. 'charges'
    sc_y = MinMaxScaler()
    y = df_transformed[target].values.reshape(-1,1)
    df_transformed[target] = sc_y.fit_transform(X=y)

    return columns_list, target, sc_x, sc_y, df_transformed



# ФУНКЦИЯ РАСЧЁТА МЕТРИК
def precision_recall_fscore(truth, preds) -> tuple:
    # Расчет True Positive, False Positive и False Negative
    tp = sum([1 for t, p in zip(truth, preds) if t == 1 and p == 1])
    fp = sum([1 for t, p in zip(truth, preds) if t == 0 and p == 1])
    fn = sum([1 for t, p in zip(truth, preds) if t == 1 and p == 0])

    # Расчет precision, recall и f1 score
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1_score




