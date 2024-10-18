# ГЕНЕРАТОР СПИСКОВ ПОЛНЫХ ФЕЙКОВЫХ ДАННЫХ 
# ЭТО СЛУЖЕБНЫЙ ФАЙЛ для упаковки сгенерированных данных в модуле 'src/utils/generator.ipynb'
# для работы файла основного проекта 'Final-project.ipynb'.
# 
# Эти Списки используются ТОЛЬКО для случайного заполнения данных людей и школ
# 
# Здесь происходит формирование списков ФЕЙКОВЫХ данных
# 

from .functions import (readJsonFile,       # чтение из json
                        readCsv2List)       # конвертация csv => list

from .listutils import (fake_file_schools,       # школы
                        fake_file_humans,        # люди (ученики и учителя)
                        fake_file_nation,        # национальность
                        fake_file_subjects,      # предметы
                        fakesrcdir               # путь записи файла
                       )

# СЛОВАРИ и СПИСКИ ФЕЙКОВЫХ ДАННЫХ
                      
fake_common_school_list: dict = readJsonFile(fakesrcdir, fake_file_schools)       # школы
fake_common_human_list: dict  = readJsonFile(fakesrcdir, fake_file_humans)        # люди (ученики и учителя)
fake_common_nation_list: list = readCsv2List(fakesrcdir, fake_file_nation)        # страны/национальность
fake_common_subjects_list: list = readCsv2List(fakesrcdir, fake_file_subjects)    # предметы

