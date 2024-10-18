# ФУНКЦИИ ДЛЯ ОБРАБОТКИ ДАННЫХ В ПРОЕКТЕ 'EDA-2'

from pandas import DataFrame, concat
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from math import ceil

from PIL import Image


# ФУНКЦИЯ СОЗДАНИЯ КАТАЛОГОВ
def mkdirs(root:str) -> None:
    # нормализация пути
    os.path.normpath(root)
    # создание каталогов
    if not os.path.isdir(root):
        os.makedirs(root)



# ФУНКЦИЯ ПОДГОТОВКИ ДАННЫХ ДЛЯ PIE ГРАФИКОВ
def groupby_tupple(df_data:DataFrame, group:list, column:str, ascending:bool =True) -> tuple:
    count_list = df_data.groupby(group)[column].count().sort_values(ascending=ascending)    # количество
    names_list:list  = count_list.index.to_list()                                           # названия
    return (count_list, names_list)



# ФУНКЦИЯ ОПРЕДЕЛЕНИЯ ДОЛИ ПАРАМЕТРОВ С БОЛЬШИМ 'std'
def percent_of_parameter(df_data:DataFrame, city:str="", subfolder:str="") -> tuple:
    # 1. Получение названий столбцов с большим отклонением 'std'

    # получение агрегаторов
    df_data = df_data.query(f'city == {city}') if city != "" else df_data
    
    df_DESCRIBE = df_data.describe()

    # выделение 'std' с сортировкой по убыванию
    df_STD = DataFrame(df_DESCRIBE.loc['std',:].sort_values(ascending=False))
    # удаление строк с малыми значениями отколонения
    std_col_list = df_STD.drop(df_STD[df_STD['std'] < 10].index).index.to_list()

    # 2. получение долей
    df_list = []
    # создание каталога
    mkdirs(root='./data/percent')
    
    for colname in std_col_list:
        # подготовка данных для распределения по осям
        x1, _ = groupby_tupple(df_data, group=[colname], column="total")

        # формирование значений по параметру и его доли с сортировкой по убыванию
        df_PERCENT = concat([x1.index.to_series(),x1, 100*x1/x1.sum()], keys=[colname,'count', 'percent'], axis=1).sort_values(by='percent', ascending=False)
        # удаление индекса
        df_PERCENT = df_PERCENT.reset_index(drop=True)

        # print(df_PERCENT.shape)

        # добавление DataFrame в список
        if df_PERCENT.shape[0]:
            df_list.append(df_PERCENT)
        else:
            std_col_list.remove(colname)

        # названия файла
        sep:str = "_" if city != "" else ""
        filename = f'./data/percent/{subfolder}/{colname.title().replace(" ","_") + sep + city}'
        df_PERCENT.to_csv(path_or_buf=os.path.normpath(filename + '.csv'), index=False)
        df_PERCENT.to_excel(os.path.normpath(filename + '.xlsx'), index=False)

    # результат список параметров + параметры DataFrame
    return (std_col_list, df_list)



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



# ФУНКЦИЯ ПОСТРОЕНИЯ ПАРНЫХ ГРАФИКОВ
def draw_pairplot(df_data:DataFrame, strtitle:str, fontsize:float =20, subfolder:str="", isShow:bool=False) -> None:
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
    


# ФУНКЦИЯ ПОСТРОЕНИЯ ГРАФИКОВ ДОЛЕЙ ДЛЯ ФИНАНСОВЫХ ПОКАЗАТЕЛЕЙ
def draw_percent(df_PERCENT, columnname:str=None, countvalue=5, isPie:bool =False, ax=plt) -> None:
    # КОПИЯ ДАННЫХ
    # без этого пункта последующий данные модифицируются
    df_PERCENT = df_PERCENT.copy()

    # выборка по количеству строк
    df_loc = df_PERCENT.loc[df_PERCENT['count'] < countvalue,'percent']

    # отладка
    # print(df_loc.sum())

    # ПОСТРОЕНИЕ ГРУППЫ ДАННЫХ ПО УСЛОВИЮ
    df_PERCENT.loc[df_PERCENT['count'] < countvalue,'percent'] = f'{df_loc.sum()} % < other'

    # df_PERCENT.loc[df_PERCENT['count'] == 2, 'percent'] = '3qty'

    if isPie:
        df_PERCENT = df_PERCENT.groupby(['percent'])['count'].sum().reset_index()
        g = ax.pie(df_PERCENT['count'], labels=df_PERCENT['percent'], autopct='%.0f%%');
    else:
        df_PERCENT = df_PERCENT.groupby(['percent', columnname])['count'].sum().reset_index()
    # g = sns.barplot(df_PERCENT, y='count', x='total', hue='percent', errorbar="ci")
    # g = sns.scatterplot(df_PERCENT, y='count', x='total', hue='percent')
        g = sns.scatterplot(df_PERCENT.sort_values(by='count', ascending=False), y='count', x=columnname, hue='percent', size='percent', style='count', palette='bright', ax=ax);
    return g




# ФУНКЦИЯ ПОСТРОЕНИЯ ГРАФИКОВ  ПАР ПАРАМЕТРОВ
def plot_box_grid(data:DataFrame, figsize=(20,20), kind:str ="violin", # вид графика
                  # параметры заголовка
                  strtitle:str="", fontsize:float=30, x_title:float=0, y_title:float=0,
                  # параметры примечания
                  strnote:str="", x_note:float=0, y_note:float=0,
                  # расстояние между графиками
                  wspace:float=0.3, hspace:float=None,
                  # параметры колонок легенды
                  legend_col:int=1, legend_fontsize:int=12,
                  # выборка строк в DataFrame
                  nrow:int =4,
                  city:str ="",
                  subfolder:str =""
                  ) -> None:
    """ФУНКЦИЯ ПОСТРОЕНИЯ ГРАФИКОВ ПАР ПАРАМЕТРОВ"""
    fig = plt.figure(figsize=figsize)

    # УСЛОВИЕ ВЫБОРА DataFrame
    if kind == "percent":
        # список колонок и базы
        col_list, df_list = percent_of_parameter(df_data=data, city=city)
    else:
        # список колонок
        col_list = data.columns.to_list()

    # количество ячеек сетки , например "2x2"
    num_cell = ceil(len(col_list)**0.5)

    # общий заголовок фигуры
    fig.suptitle(t=strtitle,
                 size=fontsize,
                #  fontdict={'fontsize':fontsize},
                 x=x_title, y=y_title
                 );    


    # заполнение фигур по заданной сетке
    # конечные ячейки без данных не отображаются
    for i in range(len(col_list)):
        # print(i)
        
        ax = plt.subplot(num_cell, num_cell, i+1);

        # виды графиков
        if kind == "violin":
            sns.violinplot(data, y=col_list[i], ax=ax);
            # ax.set_title(col_list[i] + " vs count")
            ax.axes.xaxis.set_ticklabels([]);
            ax.set_xlabel(col_list[i], fontsize=8);

        elif kind == "count":
            sns.countplot(data, y=col_list[i], ax=ax);           
            ax.set_title(col_list[i] + " vs count")
            
        elif kind == "scatter":
            x, y = groupby_tupple(data, group=[col_list[i]], column="total")
            ax = plt.scatter(x=x, y=y);
            # ax.set_title(y)
            ax.axes.set_xlabel("count");
            ax.axes.set_ylabel(col_list[i]);
        
        elif kind == "swarm":
            x, y = groupby_tupple(data, group=[col_list[i]], column="total")
            sns.swarmplot(x=x, ax=ax)

        elif kind == "scatterplot":
            g3 = sns.scatterplot(data,
                                    x=col_list[i],
                                    y='rent amount',
                                    size='city',
                                    hue='rooms',
                                    style='bathroom',
                                    palette='bright',
                                    ax=ax,
                                    legend='full'
                                    );
            
            # смещение легенды
            sns.move_legend(g3, "upper left", bbox_to_anchor=(1.05, 1), ncol=legend_col, title_fontsize=legend_fontsize)
            ax.set_title(col_list[i] + " vs rent amount")

        elif kind == "plot":
            x, y = groupby_tupple(data, group=[col_list[i]], column="total")
            
            ax.plot(x, y)
            ax.scatter(x=x, y=y);
           
            ax.set_title(col_list[i] + " vs count")
            ax.axes.set_xlabel("count");
            ax.axes.set_ylabel(col_list[i]);
        
        elif kind == "kde":
            g = sns.histplot(data=data, x=col_list[i], kde=True, color="orange", ax=ax)
            if g.lines:
                g.lines[0].set_color('crimson')
                
            ax.set_title(col_list[i] + " vs rent amount")

        elif kind == "percent":
            g = draw_percent(df_list[i], col_list[i],
                             countvalue=df_list[i].iloc[nrow,1],   # только N первых строк
                             isPie=False, ax=ax)
            
            ax.set_title(f"share of {col_list[i]}, %")
            ax.axes.set_ylabel("total");
            ax.axes.set_ylabel("count");
            g.legend(bbox_to_anchor=(1.05, 1), loc='upper left',ncol=legend_col)


        # if not kind in ["scatter","joint", "scatterplot", "plot", "kde","count", "percent"]:
            # ax.axes.xaxis.set_ticklabels([]);
            # ax.set_xlabel(col_list[i], fontsize=8);


    # Примечание
    fig.text(s=strnote, fontdict={'fontsize':14}, x=x_note, y=y_note);
    
    # расстояние между графиками
    # ax.figure.tight_layout(w_pad=0.9)
    fig.subplots_adjust(wspace=wspace, hspace=hspace)

    # сохранение графика
    saveplot(fig=fig, strtitle=strtitle, subfolder=subfolder)
    plt.show()




# ФУНКЦИЯ ПОЛУЧЕНИЯ ОТСОРТИРОВАННОГО СПИСКА СТОЛБЦОВ DataFrame
def get_sorted_unique_values(dataframe, ascending:bool =True):
    # уникальные значения
    unique_counts = dataframe.nunique()
    # сортировка
    sorted_columns = unique_counts.sort_values(ascending=ascending)
    # получение списка
    return sorted_columns.index.tolist()



# ФУНКЦИЯ ПОСТРОЕНИЯ ДОЛЕЙ НЕФИНАНСОВЫХ ПАРАМЕТРОВ
def plot_pie_grid(df_data:DataFrame, non_cash_list:list, figsize=(30,30),
                  # параметры заголовка
                  strtitle:str="", fontsize:float=30, x_title:float=0, y_title:float=0,
                  radius:float = 1,
                  labels_fontsize=12,
                #   legend_place:tuple=(1.15, 0, 0.5, 1),
                  # Note/Примечание
                  strnote:str="", x_note:float =1, y_note:float =1,
                  subfolder:str="",
                  ) -> None:
    """ФУНКЦИЯ ПОСТРОЕНИЯ ДОЛЕЙ"""
    fig = plt.figure(figsize=figsize)

    # отсортированный список колонок для оформления вывода результата
    col_list = get_sorted_unique_values(df_data[non_cash_list[:1] + non_cash_list[2:]])
    # количество ячеек сетки , например "2x2"
    num_cell = ceil(len(col_list)**0.5)

    # общий заголовок фигуры
    fig.suptitle(t=strtitle,
                 size=fontsize,
                #  fontdict={'fontsize':fontsize},
                 x=x_title, y=y_title
                 );


    # заполнение фигур по заданной сетке
    # конечные ячейки без данных не отображаются
    for i in range(len(col_list)):
        # print(i)
        
        ax = plt.subplot(num_cell, num_cell, i+1);

        # виды графиков
        # подготовка данных для распределения по осям
        x1, y1 = groupby_tupple(df_data, group=[col_list[i]], column="total", ascending=False)
        # подготовка к надписям в легенде
        key_list = x1.index.to_list()                   # ключи
        val_list = x1.to_list()                         # значения
        legend_dict = dict(zip(key_list, val_list))     # словарь для построения % в легенде
        # список долей
        percent_list = [(legend_dict[key]/sum(val_list)) * 100 for key in legend_dict]
            
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
        ax.pie(x=x1, labels=y1, radius=radius,
            # внутренние кромки
            wedgeprops={"linewidth": 1, "edgecolor": "white"},
            # надписи 'y1'
            labeldistance=1.1,
            # надписи ДОЛИ '%'
            autopct='%1.2f%%',
            # расстояние от центра
            pctdistance=0.8,
            textprops={'fontsize':labels_fontsize},
            #  explode=[0.1 for i in range(len(x1))]
            explode=explode_list,
            frame=False
            );
        
        # print([0.1 if (legend_dict[key]/sum(val_list)) * 100 > 0.12 else 0.3 for key in legend_dict])
        
        # легенда
        # список лейблов в легенде
        legend_list = [f"{key}; qty:{legend_dict[key]}; {'{:.2f}'.format((legend_dict[key]/sum(val_list)) * 100)}%" for key in legend_dict]
        # размещение легенды
        legend_place = (0.5, -0.4, 0, 2) if len(x1) > 12 else (0.5, -0.12, 0, 2)

        # сборка
        plt.legend(labels=legend_list, title=f"{col_list[i]}; qty; share",
                   # размещение вне осей
                   loc="lower center",
                   ncol=4,
                  #  loc="center left",
                   bbox_to_anchor=legend_place,
                   fontsize=14)

       
        # заголовок внутреннего графика
        ax.set_title(
                    label=f"Доля {col_list[i]}", 
                    fontdict={"fontsize":40},
                    pad=0,
                    y=1
                )
        
        # внутренний круг  
        hole = plt.Circle((0, 0), 0.65, facecolor='white', edgecolor="white")
        ax = plt.gcf().gca().add_artist(hole)


        # ax.axes.xaxis.set_ticklabels([]);
        # ax.set_xlabel(col_list[i], fontsize=8);


    # Примечание
    fig.text(s=strnote, fontdict={'fontsize':20}, x=x_note, y=y_note);

    
    # # расстояние между графиками
    # fig.tight_layout(w_pad=2)

    # сохранение графика
    saveplot(fig=fig, strtitle=strtitle, subfolder=subfolder)
    plt.show()



# *********************
# JOINTPLOT 
# *********************

# функция загрузки файла с картинкой
def load_image_pil( column_list:list, filename:str=None):
    #Начало вашего кода

    if isinstance(filename,str) and os.path.exists(filename):
        return np.array(Image.open(filename))
    else:
        print(f"{filename} does not exists")
        return None
# ФУНКЦИЯ ПОСТРОЕНИЯ ГРАФИКА 'jointplot'
def draw_jointplot(df_data:DataFrame, column_list, subfolder:str="") -> None:
  # построение графиков с сохранением
  for val in column_list:
      g = sns.jointplot(
                        x=val,
                        y="total",
                      # hue='rent amount',
                          data=df_data,
                        kind="hex",
                          # truncate=False,
                        color="m",
                        height=3,
                        marginal_ticks=True
                        );
      # заголовок
      strtitle = "total vs " + val
      plt.title(strtitle, y=1.2)
      # сохранение графика
      saveplot(fig=g, strtitle=strtitle, subfolder=subfolder)
      # не вывыодить на экран
      plt.close()


# ФУНКЦИЯ ПОЛУЧЕНИЯ ВСЕХ ФАЙЛОВ В КАТЛОГЕ
def get_dir_files(folder_path:str="./images/vs_cut/", ext:str="") -> list:

    # Проверка существования папки
    if not os.path.exists(folder_path):
        print(f"Папка {folder_path} не существует.")
        return []
    # если каталог есть
    else:
        # добавление файлов по маске в каталог
        files_list = []
        for file in os.listdir(folder_path):
            if file.endswith(ext):
                files_list.append(file)
        return files_list


# ФУНКЦИЯ ПОСТРОЕНИЯ СБОРНОГО ГРАФИКА ИЗ КАРТИНОК
def draw_grid_jointplot(column_list:list, filedir:str="./images/vs_cut/", figsize=(20,20),
                        # параметры заголовка
                        strtitle:str="", fontsize:float=30, x_title:float=0, y_title:float=0,
                        # параметры примечания
                        strnote:str="", x_note:float=0, y_note:float=0,
                        # расстояние между графиками
                        wspace:float=0.3, hspace:float=0                        
                        ) -> None:

# **************
    """ФУНКЦИЯ ПОСТРОЕНИЯ СБОРНОГО ГРАФИКА ИЗ КАРТИНОК"""
    fig = plt.figure(figsize=figsize)

    # список файлов по маске
    files_list = get_dir_files(folder_path=filedir)
    # количество ячеек сетки , например "2x2"
    num_cell = ceil(len(files_list)**0.5)


    # заполнение фигур по заданной сетке
    # конечные ячейки без данных не отображаются
    for i in range(len(files_list)):
        # print(i)
        
        ax = plt.subplot(num_cell, num_cell, i+1);

        # ЗАГРУЗКА картинок
        img_pil=load_image_pil(column_list=column_list, filename=os.path.normpath(filedir + files_list[i]))
        # распределение по осям
        ax = plt.imshow(img_pil);

        # сокрытие осей
        def cleredaxis(ax=None) -> None:
            # получение текущих осей
            ax = plt.gca() if ax is None else ax        

            # сокрытие осей
            plt.axis('off')   # убирает оси и рамку


        # очистка осей
        cleredaxis()

    # заголовок
    strtitle = strtitle
    plt.title(label=strtitle, fontdict={'fontsize':fontsize}, x=x_title, y=y_title + num_cell + 1);
    # Примечание
    fig.text(s=strnote, fontdict={'fontsize':14}, x=x_note, y=y_note);
    
    # расстояние между графиками
    # ax.figure.tight_layout(w_pad=0.9)
    fig.subplots_adjust(wspace=wspace, hspace=hspace)

    # сохранение графика
    saveplot(fig=fig, strtitle=strtitle)
    plt.show()

# **********


