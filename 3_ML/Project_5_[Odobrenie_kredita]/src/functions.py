
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

# # ФУНКЦИЯ ЗАГРУЗКИ ДАННЫХ
# def load_data(file:str) -> DataFrame:
#     return read_csv(file)

# ФУНКЦИЯ УСТАНОВКА ФОРМАТА ВЫВОДА ДЛЯ ЭЛЕМЕНТОВ СЕРИИ
def format_in_dataframe(n_format:int=2) -> None:
    n_str = "{:." + str(n_format) + "f}"
    options.display.float_format = n_str.format


# ФУНКЦИЯ ВСТАВКИ КОЛОНКИ НУМЕРАЦИИ
def set_numbercolumn(df:DataFrame, columnname:str="№", loc:int=0) -> DataFrame:
    df.insert(loc=loc, column=columnname, value=[i+1 for i in range(len(df.index))])
    return df.set_index(columnname)


# ФУНКЦИЯ ОБРАБОТКИ ДАННЫХ ТИПА 'object'
def cleared_data(df:DataFrame, include:str="object") -> DataFrame:
    column_object_list = df.select_dtypes(include=include).columns.to_list()

    for val in column_object_list:
        print(val, type(val))
        df[val] = df[val].astype("str").apply(lambda x: x.replace(" ", ""))   # замена пробела
        # df[val] = df[val].apply(lambda x: print(type(x)))   # замена пробела

    return df


# ФУНКЦИЯ ДОБАВЛЕНИЯ ОТСУТСТВУЮЩИХ СТОЛБЦОВ С НУЛЕВЫМИ ЗНАЧЕНИЯМИ В ВЫБОРКИ
def compare_and_add_columns(df1:DataFrame, df2:DataFrame) -> tuple:
    # отсутствующие столбцы по фреймам
    missing_columns_in_df1 = list(set(df2.columns) - set(df1.columns))
    missing_columns_in_df2 = list(set(df1.columns) - set(df2.columns))

    # проверка существования разницы в столбцах
    # 1. df1
    if len(missing_columns_in_df1) > 0:
        # создание нового фрейма из пропущенных столбцов со строками из оригинального фрейма
        df_delta_1 = DataFrame(columns=missing_columns_in_df1, index=df1.index)
        # заполнение столбцов нулями
        df_delta_1 = df_delta_1[missing_columns_in_df1].fillna(0)
        # добавление недостающих столбцов в конец фрейма
        df_1 = concat([df1, df_delta_1], axis=1)
    # если нет добавления столбцов
    else:
        df_1 = df1
    
    # 2. df2
    if len(missing_columns_in_df2) > 0:
        df_delta_2 = DataFrame(columns=missing_columns_in_df2, index=df2.index)
        df_delta_2 = df_delta_2[missing_columns_in_df2].fillna(0)
        df_2 = concat([df2, df_delta_2], axis=1)
    else:
        df_2 = df2    

    return df_1, df_2


# ФУНКЦИЯ ОПРЕДЕЛЕНИЯ РАЗНОСТИ В ПАРАМЕТРАХ СТОЛБЦОВ ТИПА 'object'
def diff_parameters(df_train:DataFrame, df_test:DataFrame, column_object_list:list):
    """Нужно для одинакового количества столбцов,
    которые являюся результатом кодирования"""
    diff_column_list = []
    for val in column_object_list:
        unique_values = df_train[val].unique()
        compare_values = set(unique_values).difference(set(df_test[val].unique()))
        # print(type(compare_values))
        # распаковка
        if len(compare_values):
            for val in compare_values:
                diff_column_list.append(val)                
    return diff_column_list





# ФУНКЦИЯ СОЗДАНИЯ СЛОВАРЯ С ИМЕНАМИ СТОЛБЦОВ И ИХ ЗНАЧЕНИЯМИ
def encode_features(df_train, df_test):
    """Создаем словарь с именами столбцов и их значениями"""

        # кодирование столбцов
    dummies_list = []

    df_encoded_train = df_train.copy()
    df_encoded_test = df_test.copy()
     
    encoded_columns = {}
    for col in df_train.columns:
        if df_train[col].dtype == 'object':
            encoded_columns[col] = get_dummies(df_train[col])
            # Применение метода get_dummies() к категориальному столбцу
            # df_dummies = get_dummies(df_encoded_train[col], prefix=f"{col}", dtype=int)
            df_encoded_train = get_dummies(df_encoded_train, columns=[col], prefix=f"{col}", dtype=int)

            # print(df_dummies)

            # dummies_list.append(df_dummies)

            # df_encoded_train.drop(col, axis=1, inplace=True)
        else:
            df_encoded_train[col] = df_train[col]

    # добавление в начало списка таблицы
    dummies_list.insert(0, df_encoded_train)

    for col in df_test.columns:
        if df_test[col].dtype == 'object':
            # encoded_columns[col] = get_dummies(df_train[col])
            # Применение метода get_dummies() к категориальному столбцу
            # df_dummies = get_dummies(df_encoded_train[col], prefix=f"{col}", dtype=int)
            df_encoded_test = get_dummies(df_encoded_test, columns=[col], prefix=f"{col}", dtype=int)

            # print(df_dummies)

            # dummies_list.append(df_dummies)
            # df_encoded_test.drop(col, axis=1, inplace=True)
        else:
            df_encoded_test[col] = df_test[col]

    # # разница в столбцах для train по отношению к test
    # diff_list = diff_parameters(df_train=df_train, df_test=df_test)
    # # заполнение нулями
    # if len(diff_list) > 0:
    #     df_encoded_train[diff_list] = 0


    # # сборка датасет
    # df_FINAL = concat(dummies_list, axis=1)

    # # Применяем get_dummies к тестовым данным
    # test_encoded_df = df_test.join(encoded_columns)

    missing_columns_1 = np.setdiff1d(df_encoded_train.columns.to_list(), df_encoded_test.columns.to_list())
    missing_columns_2 = np.setdiff1d(df_encoded_test.columns.to_list(), df_encoded_train.columns.to_list())

    print(missing_columns_1, missing_columns_2)

    # заполнение отсутствующих данных нулями
    if len(missing_columns_1) > 0:
        df_encoded_test[missing_columns_1] = 0

    # заполнение отсутствующих данных нулями
    if len(missing_columns_2) > 0:
        df_encoded_train[missing_columns_2] = 0

    # Возвращаем закодированные датафреймы
    return df_encoded_train, df_encoded_test




# ФУНКЦИЯ ОБРАБОТКИ ДАННЫХ ТИПА 'object'
def cleared_data(df:DataFrame) -> DataFrame:
    column_object_list = df.select_dtypes(include="object").columns.to_list()

    for val in column_object_list:
        print(val, type(val))
        df[val] = df[val].astype("str").apply(lambda x: x.replace(" ", ""))   # замена пробела
        # df[val] = df[val].apply(lambda x: print(type(x)))   # замена пробела

    return df


# ФУНКЦИЯ МАСШТАБИРОВАНИЕ ПО ГРУППАМ
def scale_data(df_data:DataFrame,             # текучая для обработки
               df_train:DataFrame,            # тренировочная таблица
               groups_int_dict:dict,           # словарь с данными на разбиение на группы
               target:str="credit_default",
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


        """"!!! Закомментированный код группирует параметры точнее,
        но уменьшает значение f1_score с 0.44 до 0.19"""
        # # распределение
        # for i in np.arange(0, ceil(max_value), step):
            
        #     # print(i, type(i), x, type(x))  # отладка

        #     # формирование меток по условию
        #     if x is np.nan:
        #         resx = "nan"
        #     # elif (x < i) and (x < step):
        #     elif x < step:
        #         resx = "less" + str(i)
        #     elif (x >= i) and (x < (max_value-step) ):
        #         resx = str(i)
        #         # break
        #     else:
        #         resx = "more" + str(ceil(max_value-step))

        
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

    # print(df_encoded)


    # кодирование столбцов
    dummies_list = []
    for col in groups_int_dict:
        # Применение метода get_dummies() к категориальному столбцу
        df_dummies = get_dummies(df_encoded[col], prefix=f"{col}", dtype=int)

        # print(df_dummies)

        dummies_list.append(df_dummies)
        # df_encoded.drop(col, axis=1, inplace=True)

    # добавление в начало списка таблицы
    dummies_list.insert(0, df_encoded)
    df_encoded.drop(columns=list(groups_int_dict.keys()), axis=1, inplace=True)
        

    # сборка датасет
    df_FINAL_0 = concat(dummies_list, axis=1)
    
    # проверка одинаковости столбцов
    if isTrain == False:
        missing_columns = np.setdiff1d(columns_list, df_FINAL_0.columns.to_list())


        # заполнение отсутствующих данных нулями
        if len(missing_columns) > 0:
            # df_FINAL[missing_columns] = 0
            df_missing = DataFrame(columns=missing_columns, index=df_FINAL_0.index)
            df_missing[missing_columns].fillna(0)

            df_FINAL = concat([df_FINAL_0, df_missing], axis=1)
        # else:
        #     df_FINAL = df_FINAL_0.copy()
    df_FINAL = df_FINAL_0.copy()

    # смещение таргет в начало таблицы
    df_FINAL.insert(0, target, df_FINAL.pop(target))

    return step_dict, df_FINAL



# ФУНКЦИЯ масштабирования
def minmax_encoder(df_data:DataFrame, target:str="credit_default") -> list:
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

