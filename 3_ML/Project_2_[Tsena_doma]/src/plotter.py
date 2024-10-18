"""БЛОК ФУНКЦИЙ ЗАДАНИЯ 'Проект 2 [Цена дома]'""";

"""ВИЗУАЛИЗАЦИЯ ДАННЫХ"""

from math import ceil

import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame

from src.bayes_isotonic import *
from src.encoders import *
from src.forest import *
from src.linear_regression import *
from src.svm import *
from src.utils import *


# ФУНКЦИЯ ВИЗУАЛИЗАЦИИ РЕГРЕССИИ
def draw_predict_scatter(models_predictions:DataFrame, df_metrics:DataFrame,
                         suptitle:str,
                         figsize:tuple=None,
                         titlesize:float=20,
                         suptitlesize:float=20,
                         y_suptitle:float=0.95,
                         ) -> None:

    # параметры фигуры и сетки
    fig = plt.figure(figsize=figsize)
    sns.set_style("darkgrid")
    sns.set_context("talk")

    fig.suptitle(t=suptitle, fontsize=suptitlesize, y=y_suptitle)

    # название колонок
    colname_list = models_predictions.columns.to_list()[1:]

    num_cell = ceil((len(colname_list)-1)**0.5)         # количество ячеек сетки , например "2x2"

    plt.tight_layout(pad=0, w_pad=0.5, h_pad=3)
    


    # формирование загловка
    def join_metrics_title(columnname:str) -> str:
        col_list = df_metrics["metrics"].to_list()

        # строка с метриками
        str_list = ["{}={}".format(val, round(number=float(df_metrics[df_metrics["metrics"] == val][columnname].values[0]),\
                                              ndigits=2)) for val in col_list]

        # строка с метриками
        return "\n".join(str_list)
                                            
    

    # заполнение фигур по заданной сетке
    # конечные ячейки без данных не отображаются
    for i in range(len(colname_list)-1):
        ax = plt.subplot(num_cell, num_cell, i+1);
        
        # график
        sns.regplot(data=models_predictions,
                        x=colname_list[0],
                        y=colname_list[i+1],
                        ax=ax
                        );

        # значения столбцов
        y_true = models_predictions[colname_list[0]]
        y_pred = models_predictions[colname_list[i+1]]

        # Заголовок над единичным графиком
        ax.set_title(join_metrics_title(columnname=[colname_list[i+1]]), fontsize=titlesize)
        
        # лимиты осей
        ax.axis([
                y_true.min(), y_true.max(),
                y_pred.min(), y_true.max(),
                ]);








# ФУНКЦИЯ ПОСТРОЕНИЯ ГРАФИКОВ МНОЖЕСТВЕННОЙ ПОЛИНОМИАЛЬНОЙ РЕГРЕССИИ
def combination_polinomial(df_train:DataFrame, df_test:DataFrame,
                           parameters:list, degree:int, interaction_only:bool=True) -> list:
    
    """ФУНКЦИЯ ПОСТРОЕНИЯ ГРАФИКОВ МНОЖЕСТВЕННОЙ ПОЛИНОМИАЛЬНОЙ РЕГРЕССИИ

        Порядок вывода: regressor / df_валидация / df_метрики

    """


    # регрессор
    multiple_polunomial_linear_regressor = train_multiple_lin_poly_regressor(data=df_train, parameters=parameters,
                                                                             degree=degree, interaction_only=interaction_only)
    # валидация
    df_multiple_polunomial_linear = multiple_polynomial_linear_validation(model=multiple_polunomial_linear_regressor, data=df_test)
    # метрики
    df_multiple_polunomial_linear_metrics = get_metrics(df_data=df_multiple_polunomial_linear)  


    # АНАЛИЗ ПРЕДСКАЗАНИЯ МЕТОДОМ ПОЛИНОМИАЛЬНОЙ ЛИНЕЙНОЙ РЕГРЕССИИ
    draw_predict_scatter(models_predictions=df_multiple_polunomial_linear,
                        df_metrics=df_multiple_polunomial_linear_metrics,
                        suptitlesize=12,
                        titlesize=10,
                        y_suptitle=1.2,
                        suptitle=f"АНАЛИЗ ПРЕДСКАЗАНИЯ МЕТОДОМ МНОЖЕСТВЕННОЙ\n ПОЛИНОМИАЛЬНОЙ ЛИНЕЙНОЙ РЕГРЕССИИ\n" +\
                                f"degree={degree}, interaction_only={interaction_only}"
    )

    return [multiple_polunomial_linear_regressor,     # регрессор
            df_multiple_polunomial_linear,            # валидация
            df_multiple_polunomial_linear_metrics,    # метрики
            ]



# ФУНКЦИЯ ПОСТРОЕНИЯ ГРАФИКОВ РЕГРЕССИИ СЛУЧАЙНОГО ЛЕСА
def combination_forest(df_train:DataFrame, df_test:DataFrame,
                           parameters:list, n_estimators:int=1, random_state:int=0) -> list:

    """ФУНКЦИЯ ПОСТРОЕНИЯ ГРАФИКОВ РЕГРЕССИИ СЛУЧАЙНОГО ЛЕСА
    
        Порядок вывода: regressor / df_валидация / df_метрики
    """
    # регрессор
    forest_regressor = train_random_forest_regressor(data=df_train, parameters=parameters, n_estimators=n_estimators, random_state=random_state)
    # валидация
    df_forest = random_forest_validation(model=forest_regressor, data=df_test)
    # метрики
    df_forest_metrics = get_metrics(df_data=df_forest)  


    # АНАЛИЗ ПРЕДСКАЗАНИЯ МЕТОДОМ ПОЛИНОМИАЛЬНОЙ ЛИНЕЙНОЙ РЕГРЕССИИ
    draw_predict_scatter(models_predictions=df_forest,
                        df_metrics=df_forest_metrics,
                        suptitlesize=12,
                        titlesize=10,
                        y_suptitle=1.2,
                        suptitle=f"АНАЛИЗ ПРЕДСКАЗАНИЯ МЕТОДОМ РЕГРЕССИИ СЛУЧАЙНОГО ЛЕСА\n" +\
                                f"n_estimators={n_estimators}, random_state={random_state}"
    )

    return [forest_regressor,     # регрессор
            df_forest,            # валидация
            df_forest_metrics,    # метрики
            ]




# *****************************
"""ДЕРЕВО ПРИНЯТИЯ РЕШЕНИЙ"""


# ФУНКЦИЯ ПОСТРОЕНИЯ ГРАФИКОВ РЕГРЕССИИ ДЕРЕВА ПРИНЯТИЯ РЕШЕНИЙ
def combination_decision_tree(df_train:DataFrame, df_test:DataFrame,
                              parameters:list, degree:int=1, random_state:int=0) -> list:

    """ФУНКЦИЯ ПОСТРОЕНИЯ ГРАФИКОВ РЕГРЕССИИ ДЕРЕВА ПРИНЯТИЯ РЕШЕНИЙ
    
        Порядок вывода: regressor / df_валидация / df_метрики
    """
    # регрессор
    decision_tree_regressor = train_decision_tree_regressor(data=df_train, parameters=parameters, degree=degree, random_state=random_state)
    # валидация
    df_decision_tree = random_forest_validation(model=decision_tree_regressor, data=df_test)
    # метрики
    df_decision_tree_metrics = get_metrics(df_data=df_decision_tree)  


    # АНАЛИЗ ПРЕДСКАЗАНИЯ МЕТОДОМ ПОЛИНОМИАЛЬНОЙ ЛИНЕЙНОЙ РЕГРЕССИИ
    draw_predict_scatter(models_predictions=df_decision_tree,
                        df_metrics=df_decision_tree_metrics,
                        suptitlesize=12,
                        titlesize=10,
                        y_suptitle=1.2,
                        suptitle=f"АНАЛИЗ ПРЕДСКАЗАНИЯ МЕТОДОМ РЕГРЕССИИ ДЕРЕВА ПРИНЯТИЯ РЕШЕНИЙ\n" +\
                                f"degree={degree}, random_state={random_state}"
    )

    return [decision_tree_regressor,     # регрессор
            df_decision_tree,            # валидация
            df_decision_tree_metrics,    # метрики
            ]



# *****************************************
"""SVM (Kernels). МЕТОД ОПОРНЫХ ВЕКТОРОВ"""


# ФУНКЦИЯ ПОСТРОЕНИЯ ГРАФИКОВ РЕГРЕССИИ МЕТОДА ОПОРНЫХ ВЕКТОРОВ
def combination_svm(df_train:DataFrame, df_test:DataFrame,
                              parameters:list, kernel="linear", degree:int=1, epsilon:float=0.1) -> list:
    """ФУНКЦИЯ ПОСТРОЕНИЯ ГРАФИКОВ РЕГРЕССИИ МЕТОДА ОПОРНЫХ ВЕКТОРОВ
    
        Порядок вывода: regressor / df_валидация / df_метрики
    """

    # регрессор
    svm_regressor = train_svm_regressor(data=df_train, parameters=parameters, kernel=kernel, degree=degree, epsilon=epsilon)
    # валидация
    df_svm = svm_validation(regressor=svm_regressor, data=df_test)
    # метрики
    df_svm_metrics = get_metrics(df_data=df_svm)  


    # АНАЛИЗ ПРЕДСКАЗАНИЯ МЕТОДОМ ПОЛИНОМИАЛЬНОЙ ЛИНЕЙНОЙ РЕГРЕССИИ
    draw_predict_scatter(models_predictions=df_svm,
                        df_metrics=df_svm_metrics,
                        suptitlesize=12,
                        titlesize=10,
                        y_suptitle=1.2,
                        suptitle=f"АНАЛИЗ ПРЕДСКАЗАНИЯ МЕТОДОМ ОПОРНЫХ ВЕКТОРОВ\n" +\
                                f"kernel={kernel}, degree={degree}, epsilon={epsilon}"
    )

    return [svm_regressor,     # регрессор
            df_svm,            # валидация
            df_svm_metrics,    # метрики
            ]



# *****************************************
"""Gaussian Naive Bayes"""

# ФУНКЦИЯ ПОСТРОЕНИЯ ГРАФИКОВ РЕГРЕССИИ Gaussian Naive Bayes
def combination_gnb(df_train:DataFrame, df_test:DataFrame, parameters:list) -> list:

    """ФУНКЦИЯ ПОСТРОЕНИЯ ГРАФИКОВ РЕГРЕССИИ Gaussian Naive Bayes
    
        Порядок вывода: regressor / df_валидация / df_метрики
    """
    # регрессор
    gnb_regressor = train_gnb_regressor(df_data=df_train, parameters=parameters)
    # валидация (функция одинаковая для линейной регрессии)
    df_gnb = linear_validation(regressors=gnb_regressor, data=df_test, important=parameters[1:], isSimple=False)
    # метрики
    df_gnb_metrics = get_metrics(df_data=df_gnb)  


    # АНАЛИЗ ПРЕДСКАЗАНИЯ МЕТОДОМ ПОЛИНОМИАЛЬНОЙ ЛИНЕЙНОЙ РЕГРЕССИИ
    draw_predict_scatter(models_predictions=df_gnb,
                        df_metrics=df_gnb_metrics,
                        suptitlesize=12,
                        titlesize=10,
                        y_suptitle=1.2,
                        suptitle=f"АНАЛИЗ ПРЕДСКАЗАНИЯ МЕТОДОМ Gaussian Naive Bayes"
    )

    return [gnb_regressor,     # регрессор
            df_gnb,            # валидация
            df_gnb_metrics,    # метрики
            ]


# *****************************************
"""Isotonic Regression"""

# ФУНКЦИЯ ПОСТРОЕНИЯ ГРАФИКОВ РЕГРЕССИИ Isotonic Regression
def combination_iso(df_train:DataFrame, df_test:DataFrame, parameters:list) -> list:
    
    """ФУНКЦИЯ ПОСТРОЕНИЯ ГРАФИКОВ РЕГРЕССИИ Isotonic Regression
    
        Порядок вывода: regressor / df_валидация / df_метрики
    """
    # регрессор
    iso_regressor = train_iso_regressor(df_data=df_train, parameters=parameters)
    # валидация
    df_iso = iso_validation(regressors=iso_regressor, data=df_test, important=parameters)
    # метрики
    df_iso_metrics = get_metrics(df_data=df_iso)  


    # АНАЛИЗ ПРЕДСКАЗАНИЯ МЕТОДОМ ПОЛИНОМИАЛЬНОЙ ЛИНЕЙНОЙ РЕГРЕССИИ
    draw_predict_scatter(models_predictions=df_iso,
                        df_metrics=df_iso_metrics,
                        suptitlesize=12,
                        titlesize=10,
                        y_suptitle=1.2,
                        suptitle=f"АНАЛИЗ ПРЕДСКАЗАНИЯ МЕТОДОМ Isotonic Regression"
    )

    return [iso_regressor,     # регрессор
            df_iso,            # валидация
            df_iso_metrics,    # метрики
            ]

