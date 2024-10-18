# ФУНКЦИИ ДЛЯ ОБРАБОТКИ ДАННЫХ В ПРОЕКТЕ 'EDA-3'

from pandas import DataFrame, read_csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from math import ceil


# ФУНКЦИЯ СОЗДАНИЯ КАТАЛОГОВ
def mkdirs(root:str) -> None:
    # нормализация пути
    os.path.normpath(root)
    # создание каталогов
    if not os.path.isdir(root):
        os.makedirs(root)



# ФУНКЦИЯ ЗАГРУЗКИ ФАЙЛОВ
def load_file(file:str) -> DataFrame:
    with open(file=file, mode="r", encoding="UTF-8") as file:
        return read_csv(filepath_or_buffer=file,        # файл
                            delimiter=",",              # разделитель данных
                            index_col=0,                # убираем пустой 1й столбец без названия
                            skipinitialspace=True)      # пропуск пробелов перед и после разделителя


# ФУНКЦИЯ СОХРАНЕНИЯ ФАЙЛА РЕЗУЛЬТАТА ПОСТРОЕНИЯ ГРАФИКА
def saveplot(fig, strtitle:str, subfolder:str="") -> None:
    """ФУНКЦИЯ СОХРАНЕНИЯ ФАЙЛА РЕЗУЛЬТАТА ПОСТРОЕНИЯ ГРАФИКА"""

    # создание каталогов
    root = os.path.normpath("./images/" + subfolder)
    mkdirs(root=os.path.normpath("./images/" + subfolder))

    # сохранение картинки на диск
    imagename = root + '/' + strtitle.title().replace(" ","_") + ".png"
    fig.savefig(imagename, format='png', bbox_inches='tight', pad_inches=0.2)
    print(f"Файл '{imagename}' сохранён")


# ФУНКЦИЯ ПРОВЕРКИ ЯВЛЯЕТСЯ ЛИ СТРОКА ЧИСЛОМ
def isNumkey(key:str, condition:str="", isValue:bool =False) -> bool:
    # if isinstance(key, str):
        value = key.replace(condition,"").replace(".","").replace(",","").strip()
        return value if isValue else value.isdigit()
    # else:
    #     return False


# ФУНКЦИЯ ОПРЕДЕЛЕНИЕ УНИКАЛЬНЫХ (АНОМАЛЬНЫХ) ЗНАЧЕНИЙ, КОТОРЫЕ ВЫДЕЛЯЮТСЯ ИЗ НОРМЫ
def get_unique_val(df_data:DataFrame, column_name:str, condition:str) -> tuple:
    """Функция определение уникальных (аномальных) значений, которые выделяются из нормы.
       1. На выходе одижается знаяение 'True' для 1го элемента кортежа.
       Это означает, что при заданном удалении из строки 'condition' остальная часть строки цифры.
       2. Вторым элементом кортежа ожидается список уникальных значений.
       Если этот список пуст - значит составные значения не найдены.
       3. Остальные элементы кортежа - списки со значениями."""
    
    # 1. ВЫБОРКА И СОРТИРОВКА ЗНАЧЕНИЙ
    # списки
    minutes_list, other_list, sep_list, uniqe_list, false_list = [], [], [], [], []

    # словарь со значениями столбца
    duration_dict = df_data[column_name].value_counts().to_dict()

    # разделение на списки
    for key in duration_dict:
        # проверка есть ли "условие" в составе значения ячейки
        # 1. заканяивается на "условие"
        if key.endswith(condition):
            # 2. часть является числом
            isNum = isNumkey(key,condition)
            false_list.append(isNum)

            # число найдено в составе фразы
            if isNum:
              minutes_list.append(key)
        # 3. если есть ещё /или это/ строка
        else:
            other_list.append(key)
            

    # 2. ПРОВЕРКА СПИСКА НА СОДЕРЖАНИЕ УНИКАЛЬНЫХ ЗНАЧЕНИЙ
    for val in other_list:
        sep_list.append(val.split(" "))

    # уникальные значения
    uniqe_list = [val for val in np.unique(sep_list) if not val.isdigit()]
    return (set(false_list), uniqe_list, other_list, minutes_list)



# ФУНКЦИИ ОПРЕДЕЛЕНИЯ НАЛИЧИЯ СЕЗОНОВ 
def get_season_x(x, isGetNumber:bool =False):

    def roundfloat(s:str):
        return round(float('{:.0f}'.format(float(s))))
    
    # print(type(x), x)
    # отсев NaN
    if isinstance(x,str) and len(x.strip()):
        # 1. СЕЗОНА НЕТ
        # проверка на число
        if isNumkey(key=x, condition='min', isValue=False):
            # это не сезон
            # но стоит проверка на определение номера
            # столбец 'serial'
            # if isGetNumber:
                # print(x)
            #     return '{:.0f}'.format(float(0))
            # else:
                return roundfloat(0)
        # 2. Сезон ЕСТЬ
        else:
            # print(x)
            # получение номера сезона
            # опция = определение является ли сезоном
            if isGetNumber:
                # да, нужен номер сезона
                # разделение составных элементов (выделение номера сезона => '7 Seasons')
                x_list = x.split(" ")
                # print(x_list[0])
                # столбец 'season'
                return roundfloat(x_list[0])
            # если просто определяется что это сезон
            else:
                # столбец 'seasons'
                return roundfloat(1)
    # NaN
    else:
        # print(type(x))
        # любой из заполняемых столбцов
        return np.nan



# ФУНКЦИЯ ВЫБОРКИ МИНИМАЛЬНЫХ / МАКСИМАЛЬНЫХ КОЭФФИЦИЕНТОВ КОРРЕЛЯЦИИ
def get_minmaxcorr(df_corr:DataFrame, quantity:int =3) -> DataFrame:

    # список коэффициентов
    corrs = [(df_corr.iloc[row,col],  # выборка данных по строкам и столбцам
             df_corr.index[row],      # название строки
             df_corr.columns[col]     # название столбца
               # проход по всем строкам и столбцам
             ) for row in range(1, len(df_corr.columns.to_list())) for col in range(row)]

    # выбор количества и сортировка выбранных элементов
    minmax = sorted(corrs, key=lambda x: x[0], reverse=True)

    # проверка параметра отображения количества элементов
    if isinstance(quantity, int) and (quantity != 0):
        # определение начала вывода данных
        minmax = minmax[:quantity] if quantity > 0 else minmax[quantity:]
    else:
        minmax = minmax

    # результат
    return DataFrame(minmax, columns=["coef", "row_name", "col_name"]).set_index('coef')



# ФУНКЦИЯ ПОЛУЧЕНИЯ СЫРОГО СПИСКА ЗНАЧЕНИЙ
def get_list_valuesfrom_column(df_data:DataFrame, column_name:str) -> list:
    return [val for val in df_data[column_name].values.tolist()]



# ФУНКЦИЯ Получение уникальных значений имён
def get_separated_cellvalues(column_name:str, df_data:DataFrame) -> tuple:
    """Пример обработки функции:            
         1. directors_list,
         2. directors_count_list,
         3. directors_index_dict,
         4. directors_dict  = get_separated_cellvalues('director')
    """
    # значения ячеек в список
    persons_list = [str(val).split(",") for val in df_data[column_name].values.tolist()]
    # persons_index_dict = {key: str(val).split(",") for key,val in enumerate(df_MOVIE[column_name].values)}
    persons_index_dict = {key: val for key,val in enumerate(df_data[column_name].values)}
   
    persons_arr = np.unique([item.strip() for sublist in persons_list for item in sublist if item.strip()], return_counts=True)

    # слияние вложенных списков
    return (np.unique([item.strip() for sublist in persons_list for item in sublist if item.strip()]).tolist(),  # уникальные знаяения
            # уникальный список имён/названий
            persons_list,
            # словарь с сырыми данными и индексами строк
            persons_index_dict,
            # словарь 'key = имя', value = список с количеством упоминаний имени/названия в фильмах/передачах и
            #                              индексом
            dict(zip(persons_arr[0],persons_arr[1]))            
            )



# ФУНКЦИЯ СОЗДАНИЯ DataFrame ИЗ СЛОВАРЯ
def create_df(dictionary:dict, columns_list:list) -> DataFrame:
    return DataFrame(list(dictionary.items()), columns=columns_list)



# ФУНКЦИЯ Преобразование dataframe в словарь
def create_dict(df_data:DataFrame, col_1:str, col_2:str) -> dict:
    return df_data.set_index(col_1)[col_2].to_dict()



# ФУНКЦИЯ ПРОВЕРКИ НАХОЖДЕНИЯ СОВПАДЕНИЙ В СЛОВАРЕ
def check_phrase(df_data:DataFrame, phrase:str, dictionary:dict):
    found_keys = []
    for key, value in dictionary.items():
        if not isinstance(key, float) and  not isinstance(value, float):
            # осеиваем 'NaN'
            if isinstance(dictionary[key],str):
                # преобразование массива в строку
                # value = ','.join(map(str, value))
                # print(value)
                if phrase in value:
                    # список с 'id_show' для искомого названия/имени
                    found_keys.append(df_data.loc[df_data.index == key, 'show_id'].values[0])

    # return found_keys if len(found_keys) else False
    return ",".join(map(str,found_keys)) if len(found_keys) else False



# ФУНКЦИЯ ПРОВЕРКИ НАХОЖДЕНИЯ СОВПАДЕНИЙ В СЛОВАРЕ ДЛЯ МАССИВА
def check_phrase_np2(phrase:str, dictionary:dict) -> str:
    found_keys = []
    for key in dictionary:
        if not isinstance(key, float) and  not isinstance(dictionary[key], float):
            # print(type(key), key)
            # преобразование массива в строку
            value = ','.join(map(str, dictionary[key]))
            
            # осеиваем 'NaN'
            if phrase in value:
                # список с 'id_show' для искомого названия/имени
                found_keys.append(key)

    return ",".join(map(str,found_keys)) if len(found_keys) else False



# ФУНКЦИЯ установка индексирования столбца 'id'
def set_id(df_data:DataFrame, id_name:str ="id") -> DataFrame:
    return df_data.insert(loc=0, column=id_name, value=[i+1 for i in range(len(df_data.index))])



# ФУНКЦИЯ СОЗДАНИЯ СЛОВАРЯ С 'show_id:id' ДЛЯ ГРУППЫ
def create_dict_id(where_dictionary:dict, whhat_list:list) -> dict:
    # словарь с данными по группе с привязкой к 'id_show'
    # key = 'id_show'
    # value = 'id' внутри категории
    return {val: check_phrase_np2(phrase=val, dictionary=where_dictionary)\
            for val in map(str,whhat_list) if check_phrase_np2(phrase=val, dictionary=where_dictionary)}



# ФУНКЦИЯ ПОДГОТОВКИ ДАННЫХ ДЛЯ ПОСТРОЕНИЯ ГРАФИКОВ 'pie'
def get_datafor_pie(df_data:DataFrame, column_name:str='count',
                    upper:int =10,  # выбора строк (верх/низ таблицы = +/-)                    
                    kind:str=">"    # вид условия больше/меньше ....
                    ) -> tuple:
    # выборка
    x = df_data.dropna().sort_values(by=column_name, ascending=False).dropna().query(f'{column_name} {kind} {upper}')
    # print(x)
    # параметры
    labels  = x['name']
    counts  = x['count']
    percent = x['percent']

    x = x[column_name]

    # поэлементное слияние списков
    if kind == "count":
        map_list = list(map(str,zip(labels,list(map(str,zip(counts, percent))))))
    else:
        map_list = list(map(str,zip(labels,list(map(str,zip(percent, counts))))))

    # строка надписи на графике
    combined = [val.replace("'","").replace('(',"").replace(")","").replace(",",";") for val in map_list]

    return (x, labels, counts, percent, combined)


# ФУНКЦИЯ ПОСТРОЕНИЯ ПАРНЫХ ГРАФИКОВ
def draw_pairplot(df_list:list, strtitle:list, subfolder:str="pairplot",
                  kind:str="scatter",
                  hue='count',
                  isDropna:bool=True) -> None:
    i = 0
    for df_data in df_list:        
        
        if isDropna:
            df_data = df_data.dropna(axis=0, how='any')

        g = sns.pairplot(df_data,
                         corner=True,
                          dropna=isDropna,
                          hue=hue, height=3,
                         
                         palette="bright",
                         kind=kind
                         );
        g.fig.suptitle(strtitle[i].upper(), ha="center");
        # сохранение графика
        saveplot(fig=g, strtitle=strtitle[i], subfolder=subfolder)
        plt.show();
        i += 1



# ФУНКЦИЯ СЛОВАРЬ С ДОЛЯМИ В ПРОЦЕНТАХ по условию
def create_percent_dict(dictionary:dict, threshold:float, isOther:bool=False):
    # Суммируем все значения словаря
    total = sum(dictionary.values())

    percent_dict = {}
    other_sum = 0

    for key, value in dictionary.items():
        # Вычисляем процент для каждого элемента
        percent = (value / total) * 100
        if percent >= threshold:
            percent_dict[key] = percent
        else:
            other_sum += value

    if isOther:
        # Добавляем "other", если сумма меньше порогового значения
        if other_sum > 0:
            percent_dict["other"] = (other_sum / total) * 100

    return percent_dict




# ФУНКЦИЯ СЛОВАРЬ С ДОЛЯМИ В ПРОЦЕНТАХ
def dict_to_percentage(dictionary:dict):
    total = sum(dictionary.values())
    percentage_dict = {}

    for key, value in dictionary.items():
        percentage_dict[key] = (value / total) * 100

    return percentage_dict


# ФУНКЦИЯ ПОСТРОЕНИЯ ДОЛЕЙ НЕФИНАНСОВЫХ ПАРАМЕТРОВ
def plot_pie_grid_1(df_data:DataFrame, col_list:list, figsize=(30,30),
                  # параметры заголовка
                  strtitle:str="", fontsize:float=30, x_title:float=None, y_title:float=None,
                  radius:float = 1,
                  labels_fontsize=12,
                #   legend_place:tuple=(1.15, 0, 0.5, 1),
                  # Note/Примечание
                  strnote:str="", x_note:float =1, y_note:float =1, notesize:float=40,
                  subfolder:str="",
                  threshold_percent:float =1,
                  ncol_legend:int =2,
                  titlesize:float =40,
                  w_pad:float=2, h_pad:float=2,
                  isNamesinLabels:bool =False,
                  isOther:bool=True,
                  ) -> None:
    """ФУНКЦИЯ ПОСТРОЕНИЯ ДОЛЕЙ"""
    fig = plt.figure(figsize=figsize)

    # отсортированный список колонок для оформления вывода результата
    # col_list = get_sorted_unique_values(df_data[non_cash_list[:1] + non_cash_list[2:]])
    # количество ячеек сетки , например "2x2"
    num_cell = ceil(len(col_list)**0.5)

    # общий заголовок фигуры
    fig.suptitle(t=strtitle,
                 size=fontsize,
                #  fontdict={'fontsize':fontsize},
                 x=x_title, y=y_title
                 );

    # копия базы
    df_data = df_data.copy()

    # заполнение фигур по заданной сетке
    # конечные ячейки без данных не отображаются
    for i in range(len(col_list)):
        # print(i)
        
        ax = plt.subplot(num_cell, num_cell, i+1);

        # виды графиков
        # подготовка данных для распределения по осям
        x1 = df_data[col_list[i]].value_counts()
        y1 = [key for key in x1.to_dict()]

        # подготовка к надписям в легенде с разбивкой на группы по условию ниже задонного процента
        small_groups = x1[x1 < (threshold_percent / 100 * len(df_data))]
        
        # Создание объединенной группы
        df_data['newcol'] = df_data[col_list[i]].where(~df_data[col_list[i]].isin(small_groups.index), 'other')

        
        # Получение количества значений в объединенной группе
        group_counts = df_data['newcol'].value_counts()
        # val_list = val_list.to_list()                         # значения

        if isNamesinLabels:
            y1 = group_counts.values
        else:
            y1 = group_counts.index

        # словарь долей, с группировкой по долям
        legend_dict = create_percent_dict(group_counts.to_dict(), 1, isOther=isOther)
        percent_list = [val for val in legend_dict.values()]

        # print(len(percent_list), len(group_counts))

        # print("**",i, percent_list)
            
        # заполнение списка значений для выделения сегментов
        explode_list = []
        n = 0
        for val in percent_list:
            if val >= 1:
                explode_list.append(0.1)
            elif val < 1 and val > 0.5:
                explode_list.append(n+0.15)
                n += 0.12
            else:
                explode_list.append(n+0.2)
                n += 0.22
        
        
        # график
        ax.pie(x=group_counts, 
               labels=y1,
               radius=radius,
                # внутренние кромки
                wedgeprops={"linewidth": 1, "edgecolor": "white"},
                # надписи 'y1'
                labeldistance=1.1,
                # надписи ДОЛИ '%'
                autopct='%1.2f%%',
                # расстояние от центра
                pctdistance=0.8,
                textprops={'fontsize':labels_fontsize},
                explode=explode_list,
                frame=False
                );
        
        # легенда
        # список лейблов в легенде
        legend_list = [f"{key}; qty: {group_counts[key]}; {'{:1.2f}'.format(legend_dict[key])}%" for key in legend_dict]
        # размещение легенды
        legend_place = (0.5, -0.4, 0, 2) if len(group_counts) > 12 else ((0.5, -0.2, 0, 2) if len(group_counts) < 8 else (0.5, -0.4, 0, 2))

        # сборка
        plt.legend(labels=legend_list, title=f"{col_list[i]}; share",
                   # размещение вне осей
                   loc="lower center",
                   ncol=ncol_legend,
                  #  loc="center left",
                   bbox_to_anchor=legend_place,
                   fontsize=14)
       
        # заголовок внутреннего графика
        ax.set_title(
                    label=f"Доля '{col_list[i]}'", 
                    fontdict={"fontsize":titlesize},
                    pad=0,
                    y=1
                )
        
        # внутренний круг  
        hole = plt.Circle((0, 0), 0.65, facecolor='white', edgecolor="white")
        ax = plt.gcf().gca().add_artist(hole)

    # Примечание
    fig.text(s=strnote, fontdict={'fontsize':notesize}, x=x_note, y=y_note);

    
    # расстояние между графиками
    fig.tight_layout(w_pad=w_pad, h_pad=h_pad)

    # сохранение графика
    saveplot(fig=fig, strtitle=strtitle, subfolder=subfolder)
    plt.show()

    # return group_counts



# ФУНКЦИЯ ПОЛУЧЕНИЯ ЗНАЧЕНИЕ ИНДЕКСА СТРОКИ С НАИМЕНЬШИМ ПРОЦЕНТОМ
def get_percent_value(df_data:DataFrame, column_name:str ="percent", threshold_row:int=5) -> float:
        # значение индекса строки с наименьшим процентом
        min_index = df_data.dropna().sort_values(by=column_name, ascending=False).iloc[:threshold_row,:].index[-1]
        # print(min_index)
        # значение ячейки
        return df_data[column_name].iloc[min_index]



# ФУНКЦИЯ ПОСТРОЕНИЯ ДОЛЕЙ НЕФИНАНСОВЫХ ПАРАМЕТРОВ
def plot_pie_grid_2(df_list:list, col_list:list, figsize=(30,30),
                  # параметры заголовка
                  strtitle:str="", fontsize:float=30, x_title:float=None, y_title:float=None,
                  radius:float = 1,
                  labels_fontsize=12,
                  legend_place:tuple=(1.15, 0, 0.5, 1),
                  ncol_legend:int =2,
                  # Note/Примечание
                  strnote:str="", x_note:float =0.1, y_note:float =0.01, notesize:float=40,
                  subfolder:str="",
                  threshold_percent:float =1,
                  titlesize:float =40,
                  w_pad:float=2, h_pad:float=2,
                # количество строк для выборки (верх/низ значения)
                  threshold_percent_rows:int=5,
                  column_name:str="percent",
                  kind:str=">"
                #   percent_list:list=[]
                  ) -> None:
    
    """ФУНКЦИЯ ПОСТРОЕНИЯ ДОЛЕЙ"""
    fig = plt.figure(figsize=figsize)

    # отсортированный список колонок для оформления вывода результата
    # col_list = get_sorted_unique_values(df_data[non_cash_list[:1] + non_cash_list[2:]])
    # количество ячеек сетки , например "2x2"
    num_cell = ceil(len(col_list)**0.5)

    # общий заголовок фигуры
    fig.suptitle(t=strtitle,
                 size=fontsize,
                #  fontdict={'fontsize':fontsize},
                 x=x_title, y=y_title
                 );

    # # копия базы
    # df_data = df_data.copy()

    # заполнение фигур по заданной сетке
    # конечные ячейки без данных не отображаются
    for i in range(len(col_list)):
        # print(i)

        # # копия базы
        # df_data = df_list[i].copy()
        
        ax = plt.subplot(num_cell, num_cell, i+1);

        # виды графиков
        # # подготовка данных для распределения по осям
        # x1 = df_data[col_list[i]].value_counts()
        # y1 = [key for key in x1.to_dict()]

        # upper = df_list[i]['percent'].value_counts(sort=True, ascending=False).to_list()[:threshold_percent_rows]
        upper = df_list[i].dropna().sort_values(by=column_name, ascending=False )[column_name].to_list()[:threshold_percent_rows]
        # print(upper)

        # значение индекса строки с наименьшим процентом
        percent_value = get_percent_value(df_data=df_list[i],
                                          column_name=column_name,
                                          threshold_row=threshold_percent_rows)

        # print(percent_value)

        x1, y1, counts_list, percent_list, combined_list = get_datafor_pie(df_data=df_list[i],
                                                                           column_name=column_name,
                                                                        #    upper=min(upper)
                                                                           upper=percent_value,
                                                                           kind=kind
                                                                           )
        

        # словарь долей, с группировкой по долям
        legend_dict = create_percent_dict(x1.to_dict(), 1)

        if col_list[i] == 'director':
            # заполнение списка значений для выделения сегментов
            explode_list = []
            n = 0
            for val in percent_list:
                if float(val) >= 1:
                    explode_list.append(0.1)
                elif float(val) < 1 and float(val) > 0.5:
                    # explode_list.append(n+0.15)
                    # n += 0.12
                    explode_list.append(n+0.15)
                    n += 0.1
                else:
                    # explode_list.append(n+0.2)
                    # n += 0.22
                    explode_list.append(n+0.2)
                    n += 0.22
        else:
            explode_list = [0 for _ in range(len(x1))]
            
        
        # график
        ax.pie(x=x1, 
               labels=combined_list,
               radius=radius,
                # внутренние кромки
                wedgeprops={"linewidth": 1, "edgecolor": "white"},
                # надписи 'y1'
                labeldistance=1.1,
                # надписи ДОЛИ '%'
                autopct='%1.2f%%',
                # расстояние от центра
                pctdistance=0.8,
                textprops={'fontsize':labels_fontsize},
                # explode=explode_list,
                frame=False
                );
        
        # легенда
        # список лейблов в легенде
        legend_list = [f"{key}; {'{:1.3f}'.format(x1[key])}%; qty: {'{:.0f}'.format(legend_dict[key])}" for key in legend_dict]
        # размещение легенды
        legend_place = (0.5, -0.4, 0, 2) if len(x1) > 12 else ((0.5, -0.2, 0, 2) if len(x1) < 8 else (0.5, -0.4, 0, 2))

        # сборка
        plt.legend(labels=legend_list, title=f"{col_list[i]}; share",
                   # размещение вне осей
                   loc="lower center",
                   ncol=ncol_legend,
                  #  loc="center left",
                   bbox_to_anchor=legend_place,
                   fontsize=14)
       
        # заголовок внутреннего графика

        strforlabel = f"{'{:.3f}'.format(percent_value)}% ({'{:.0f}'.format(counts_list.min())} фильмов)"
        # if kind == "count":
        #     strforlabel = f"{'{:.3f}'.format(percent_value)}% ({'{:.0f}'.format(counts_list.min())} фильмов)"
        # else:
        #     strforlabel = f"{'{:.3f}'.format(counts_list.min())}% ({'{:.0f}'.format(percent_value)} фильмов)"

        ax.set_title(
                    label=f"Лидеры среди '{col_list[i]}', доля {kind} {strforlabel}", 
                    fontdict={"fontsize":titlesize},
                    pad=0,
                    y=1
                )
        
        # внутренний круг  
        hole = plt.Circle((0, 0), 0.65, facecolor='white', edgecolor="white")
        ax = plt.gcf().gca().add_artist(hole)

    # Примечание
    fig.text(s=strnote, fontdict={'fontsize':notesize}, x=x_note, y=y_note);

    
    # расстояние между графиками
    fig.tight_layout(w_pad=w_pad, h_pad=h_pad)

    # сохранение графика
    saveplot(fig=fig, strtitle=strtitle, subfolder=subfolder)
    plt.show()


# ФУНКЦИЯ ПОСТРОЕНИЯ ПАРНЫХ ГРАФИКОВ
def draw_pairplot_2(df_data:DataFrame, strtitle:str, fontsize:float =20, subfolder:str="", isShow:bool=False) -> None:
    g = sns.pairplot(df_data, corner=True, kind='reg',
                     # цвет графиков
                     plot_kws = {'color': 'green', 'marker': 'o',       # основные
                                 # линия регрессии
                                 'line_kws':{'color':'red'},                                 
                                 # данные наполнения 'scatterplot'
                                 'scatter_kws': {'edgecolor': 'white',
                                                 'alpha': 0.1}},     
                     # диагональ
                     diag_kws = {'color': 'orange'},
                    );
    # заголовок
    g.fig.suptitle(strtitle, y=0, fontsize=60);

    # сохранение файла
    saveplot(fig=g, strtitle=strtitle, subfolder=subfolder)

    # режим отображения графиков на экране
    if isShow:
        plt.show()
    else:
        plt.close()
    