"""БЛОК ФУНКЦИЙ ЗАДАНИЯ 'Проект 2 [Цена дома]'""";

"""ОБРАБОТКА ДАННЫХ"""

from pandas import read_csv,  DataFrame,  concat
import numpy as np

from sklearn.preprocessing import StandardScaler,  LabelEncoder
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics

METRICS_LIST = ["MSE", "RMSE", "MAE", "R2", "Adjusted_R2"]


# ФУНКЦИЯ ЗАГРУЗКИ ДАННЫХ
def load_data(file:str) -> DataFrame:
    return read_csv(file)

# ФУНКЦИЯ ОПРЕДЕЛЕНИЯ КОЛИЧЕСТВА И НАЗВАНИЯ ПУСТЫХ ЗНАЧЕНИЙ
def get_nan_cells(df_data:DataFrame) -> DataFrame:

    nan_info = df_data.isna().sum().sort_values(ascending=False)
    nan_dict = nan_info.to_dict()
    return {key: val for key, val in nan_dict.items() if val is not None and val != 0}


# ФУНКЦИЯ ОПРЕДЕЛЕНИЯ СТРОК, ГДЕ ТОЛЬКО ЗНАЯЕНИЯ 'NaN'
def get_nan_all(df_data:DataFrame) -> DataFrame:
    result = df_data.loc[df_data.isna().all(axis=1)]
    print(f"Всего строк в выборке: {len(df_data)}",
          f"Потеряно строк: {0 if len(result) == 0 else len(df_data) - len(result)}",
          f"Новое прогнозируемое количество строк: {len(df_data) - len(result)}",
          sep="\n")
    return result


# ФУНКЦИЯ ОПРЕДЕЛЕНИЯ СТРОК, ГДЕ ВСТРЕЧАЮТСЯ ЗНАЯЕНИЯ 'NaN'
def get_nan_any(df_data:DataFrame) -> DataFrame:
    result = df_data.loc[df_data.isna().any(axis=1)]
    print(f"Всего строк: {len(df_data)}",
          f"Потеряно строк: {len(result) if len(result) == 0 else len(df_data) - len(result)}",
          f"Новое прогнозируемое количество строк: {len(df_data) - len(result)}",
          sep="\n")
    return result


# ФУНКЦИЯ ОПРЕДЕЛЕНИЯ КОЛИЧЕСТВА И НАЗВАНИЯ ПУСТЫХ ЗНАЧЕНИЙ
def get_missing_data(df_data:DataFrame) -> DataFrame:
    # словарь количества и названия пустых значений
    nan_columns_dict = get_nan_cells(df_data=df_data)

    print(f"Доля отсутсвующих значений 'NaN' в исходных данных по {len(nan_columns_dict)} столбцам")
    print("Всего строк: ", df_data.shape[0])
    # print(nan_columns_dict)

    columns_list = ["nan_rows", "data_percent", "missing_percent", "datatype"]

    df_result = DataFrame(data={key: [val,                                      # nan_rows - количество отсутствующих данных (строк)
                                 100 * round(1 - val / df_data.shape[0], 5),    # data_percent - данные есть, в процентах
                                 100 * round(val / df_data.shape[0], 5),        # missing_percent - то же, в процентах                                       
                                 df_data[key].dtypes                                      # тип данных
                                ] for key, val in nan_columns_dict.items()
                          },
                                index=columns_list
                        ).T
    
    # приведение типов для цифровых столбцов
    df_result[columns_list[:-1]] = df_result[columns_list[:-1]].astype(np.float64)
    df_result[columns_list[-1]] = df_result[columns_list[-1]].astype(str)

    return df_result,\
           list(nan_columns_dict.keys())  # названия столбцов



# СОЗДАНИЕ ТАБЛИЦЫ СТРАНЕНИЯ МЕТРИК
def df_metrics_construct(metrics_dict:dict, metrics_list:list=METRICS_LIST) -> DataFrame:
    
    # словарь таблиц с метриками
    df_metrics_dict = {}
    df_result_metrics_dict = {}

    for i in range(len(metrics_list)):
        df_metric = DataFrame()
        df_metric['simple'] = [metrics_dict['simple'][0].T[i].values[1:].min()]
        df_metric['multiple'] = [metrics_dict['multiple'][0].T[i].values[1:].min()]

        pol_keys = metrics_dict['polinomial'].keys()
        # print(pol_keys)

        for degree_n in pol_keys:
            df_metric[f'polinomial_degree_{degree_n}'] = [metrics_dict['polinomial'][degree_n].T[i].values[1:].min()]

        forest_keys = metrics_dict['forest'].keys()
        # print(forest_keys)

        for val in forest_keys:
            df_metric[f'forest_{val}'] = [metrics_dict['forest'][val].T[i].values[1:].min()]

        tree_keys = metrics_dict['tree'].keys()
        # print(tree_keys)

        for val in tree_keys:
            df_metric[f'tree_{val}'] = [metrics_dict['tree'][val].T[i].values[1:].min()]

        svm_keys = metrics_dict['svm'].keys()
        # print(tree_keys)

        for val in svm_keys:
            df_metric[f'svm_{val}'] = [metrics_dict['svm'][val].T[i].values[1:].min()]
            
        
        df_metric['gnb'] = metrics_dict['gnb'][0].T[i].values[1:].min()
        df_metric['iso'] = metrics_dict['iso'][0].T[i].values[1:].min()

    
        df_metrics_dict[metrics_list[i]] = df_metric
    
    for key in df_metrics_dict:
        # СРАВНЕНИЕ метрик
        df_metric_T = df_metrics_dict[key].T
        df_metric_T.columns=[key]
        df_result_metrics_dict[key] = df_metric_T.sort_values(by=key, ascending=True)


    # список DataFrame из значений словаря
    return concat([df for df in df_result_metrics_dict.values()], axis=1)

