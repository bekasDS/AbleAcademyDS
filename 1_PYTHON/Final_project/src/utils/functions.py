# ЭТО СЛУЖЕБНЫЙ ФАЙЛ для чтения фейковых данных из файлов
# ФУНКЦИИ
import json
# import csv

# ФОРМИРОВАНИЕ СЛОВАРЯ ИЗ ФАЙЛА С 'JSON' ДАННЫМИ
def readJsonFile(mydir, filename) -> dict:
    common_dict = {}  # словарь с общими данными
    fpath = mydir + "/" + filename
    # заполнение словаря
    with open(fpath, "r", encoding="UTF-8") as ofile:
        common_dict = json.load(ofile)
    # результат
    return common_dict



# ФОРМИРОВАНИЕ СПИСКА ИЗ ФАЙЛА 'CSV' ДАННЫМИ
def readCsv2List(mydir, filename) -> list:
    
    # common_list = []  # список с общими данными


    fpath = mydir + "/" + filename
    # заполнение словаря
    with open(fpath, "r", encoding="UTF-8") as ofile:
        # common_list = csv.reader(ofile)
        common_list: str = ofile.readline().split(",")
        # common_list = common_list.split(",")
    # результат
    return common_list


