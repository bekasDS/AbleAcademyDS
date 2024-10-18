# КОНСТАНТЫ
# в том файле задаются константы для работы программ =>
#   1. генератор фейковых списков
#   2. создание объектов классов
# 
# Пороги значений заданы произвольно.
# При необходимости константы можно изменить на необходимые значения
# 
# Остальные данные задаются непосредственно в генераторе './src/utils/generator.ipynb'

# путь к файлу
fakesrcdir = "src/files"

# названия файлов
fake_file_humans   = "humans.json"     # люди (ученики и учителя)
fake_file_schools  = "schools.json"    # школы
fake_file_nation   = "nation.csv"      # национальность
fake_file_subjects = "subjects.csv"    # предметы

# возраст ОТ/ДО
# ученики
agestud_1: int = 10
agestud_2: int = 17
# учителя
agetchr_1: int = 28
agetchr_2: int = 65

# количество ОТ/ДО
# ученики
qustud_1: int = 10
qustud_2: int = 20
# учителя
qutchr_1: int = qustud_1 // 6
qutchr_2: int = qustud_2 // 2

# статус
status_student = "ученик"
status_teacher = "учитель"

# ЗАГОЛОВКИ СПИСКОВ В ФАЙЛАХ 'csv'
# заголовок таблицы школ
head_school_list = ["id",
                    "Школа",
                    "Адрес",
                    "Телефон",
                    "e-mail",
                    "Учеников_max",
                    "Учителей_max",
                    "Всего_Учителей",
                    "Всего_Учеников"
                    ]


# заголовок таблицы учеников и учителей
head_humans_list = ["id",
                    "Статус",
                    "Имя",
                    "Фамилия",
                    "Возраст",
                    "Пол",
                    "Гражданство",
                    # "Школа",
                    "Предметы"
                    ]

