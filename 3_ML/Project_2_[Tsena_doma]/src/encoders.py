"""БЛОК ФУНКЦИЙ ЗАДАНИЯ 'Проект 2 [Цена дома]'""";

"""КОДИРОВАНИЕ""";


from pandas import DataFrame, Series
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.utils import METRICS_LIST



# ФУНКЦИЯ МАСШТАБИРОВАНИЯ МОДЕЛИ
def create_scaled_linear_regression_model(X_train:Series, y_train:Series):
    # Создание объекта масштабирования данных
    scaler = StandardScaler()
    
    # Масштабирование признаков обучающей выборки
    X_scaled = scaler.fit_transform(X_train)
    
    # Создание модели линейной регрессии
    model = LinearRegression()
    
    # Обучение модели на масштабированных данных
    model.fit(X_scaled, y_train)
    
    return model, scaler




# ФУНКЦИЯ стандартизации с трансформацией массива
def standardscaler_encoder(df_data:DataFrame) -> list:
    # стандартизация с трансформацией массива

    df_transformed = df_data.copy()

    columns_list = df_transformed.columns.to_list()

    important = columns_list.pop(columns_list.index("SalePrice")) if "SalePrice" in columns_list else columns_list

    # объект масштабирования
    # 1. все столбцы
    sc_x = StandardScaler()
    X = df_transformed[important].values.reshape(-1,1)
    df_transformed[important] = sc_x.fit(X=X).transform(X=X)

    # 2. 'SalePrice'
    sc_y = StandardScaler()
    y = df_transformed["SalePrice"].values.reshape(-1,1)
    df_transformed["SalePrice"] = sc_y.fit(X=y).transform(X=y)

    return sc_x, sc_y, df_transformed


# ФУНКЦИЯ КОДИРОВАНИЯ 'Linear Encoding'
def label_encoding_sklearn(df_data:DataFrame) -> tuple:
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(df_data.values.reshape(-1,1).ravel())
    return (encoded, encoder)


# ФУНКЦИЯ ПОСТРОЕНИЯ ЗАКОДИРОВАННЫХ ДАННЫХ ИЗ 'DataFrame'
def get_encoded_data(df_data:DataFrame,
                            #   функция кодирования
                              function=label_encoding_sklearn,
                              isTrain=True
                              ) -> dict:
    
    df_encoded_dict = {}                # словарь закодированных значений
    # df_data_encoded = df_data.copy()    # закодированные данные

    # стандаризаторы данных
    sc_x, sc_y, df_data_encoded = standardscaler_encoder(df_data=df_data)
   
    # закодированные данные
    for val in df_data.columns.to_list():
        encoded, encoder = function(df_data=df_data_encoded[val])    # данные, кодировщик
        df_encoded_dict[val] = [encoded, encoder]
        df_data_encoded[val] = encoded

    return  [sc_x, sc_y, df_encoded_dict, df_data_encoded] if isTrain else [df_encoded_dict, df_data_encoded]


"""МЕТРИКИ"""

# ФУНКЦИЯ РАСЧЁТА МЕТРИК ОШИБОК ПРЕДСКАЗАНИЙ ВЫБОРКИ
def get_metrics(df_data:DataFrame, k:int=1):

    df_metrics = DataFrame()
    targets = df_data['SalePrice']

    for colname in df_data.columns.to_list()[2:]:
        predictions  = df_data[colname]
        mse_sklearn  = mean_squared_error(targets, predictions)
        rmse_sklearn = mean_squared_error(targets, predictions, squared=False)
        mae_sklearn  = mean_absolute_error(targets, predictions)
        r2_sklearn   = r2_score(targets, predictions)

        n, k = len(targets), 1
        adjusted_r2_sklearn = 1 - (1 - r2_score(targets,predictions)) * ((n - 1) / (n - k - 1))

        df_metrics["metrics"] = METRICS_LIST
        df_metrics[colname] = [mse_sklearn, rmse_sklearn, mae_sklearn, r2_sklearn, adjusted_r2_sklearn]

    return df_metrics




