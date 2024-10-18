# ОРГАНИЗАЦИИ

# условия запуска модуля при import и самостоятельном
if __name__ != "__main__":
    
    # константы
    from src.utils.listutils import (head_humans_list,
                                     head_school_list)
    # классы
    from .users import *


# работа с путями
from os import makedirs
from os.path import isdir


# УЧЕБНОЕ ЗАВЕДЕНИЕ
class School:

    # счётчик класса
    # счётчики конструкторов
    count = 0
    schools_dict = dict() # словарь с данными школ

    # путь к записи файлов
    mydir = "./csv"

    def __init__(self,
                 name: str,             # название учебного заведения
                 address: str,          # адрес
                 phone: str,            # телефон
                 email: str,            # электронная почта
                 students_count: int,   # максимальное число учащихся
                 teachers_count: int    # максимальное число учеников
                 ) -> None:
        
        # данные по людям и школам
        self.common_school_info_dict = dict()   # основной словарь данных по школе и людям
        self.stud_school_list = []              # ученики
        self.teach_school_list = []             # учителя

        
        # ПРОВЕРКА на соответствие типов данных родительского класса
        self.__name = name if type(name) is str else print(check_errortype(name,"str"))                                          # имя
        self.__address = address if type(address) is str else print(check_errortype(address,"str"))                              # адрес
        self.__phone = phone if type(phone) is str else print(check_errortype(phone,"str"))                                      # телефон
        self.__email = email if type(email) is str else print(check_errortype(email,"str"))                                      # электронная почта
        self.__students_count = students_count if type(students_count) is int else print(check_errortype(students_count,"int"))  # количество студентов
        self.__teachers_count = teachers_count if type(teachers_count) is int else print(check_errortype(teachers_count,"int"))  # количество учителей

        self.__id: int = __class__.count   # id школы

        __class__.count += 1    # счётчик объектов

        
        # общий словарь созданных объектов школ (доступ на уровне класса по всем школам)
        __class__.schools_dict[self.__name] = {"id": self.__id,                                # идентификатор
                                               "name"   : self.__name,                  # название школы
                                               "address": self.__address,               # адрес
                                               "phone"  : self.__phone,                 # телефон
                                               "email"  : self.__email,                 # электронная почта
                                               "students_count": self.__students_count, # МАКСИМАЛЬНОЕ количество учащихся
                                               "teachers_count": self.__teachers_count  # МАКСИМАЛЬНОЕ количество учителей
                                               }



        # данные по текущей школе
        self.common_school_info_dict = {"info": {"id": str(self.__id),                           # идентификатор
                                                 "name"   : self.__name,                         # название школы
                                                 "address": self.__address,                      # адрес
                                                 "phone"  : self.__phone,                        # телефон
                                                 "email"  : self.__email,                        # электронная почта
                                                 "students_count": str(self.__students_count),   # МАКСИМАЛЬНОЕ количество учащихся
                                                 "teachers_count": str(self.__teachers_count)    # МАКСИМАЛЬНОЕ количество учителей
                                                }
        }

    # ДОСТУП К АТРИБУТАМ
    @property
    def get_id(self):
        return self.__id
    # название школы
    @property
    def get_name(self):
        return self.__name

    @property
    # изменение адреса
    def get_address(self):
        return self.__address

    @property
    # телефон
    def get_phone(self):
        return self.__phone

    @property
    # электронная почта
    def set_email(self):
        return self.__email



    # ИЗМЕНЕНИЕ АТРИБУТОВ
    # изменение названия школы
    def set_name(self, name: str) -> str:
        self.__name = name if type(name) is str else print(check_errortype(name,"str"))

    # изменение адреса
    def set_address(self, address: str) -> str:
        self.__address = address if type(address) is str else print(check_errortype(address,"str"))

    # изменение телефона
    def set_phone(self, phone: str) -> str:
        self.__phone = phone if type(phone) is str else print(check_errortype(phone,"str"))

    # изменение электронной почты
    def set_email(self, email: str) -> str:
        self.__email = email if type(email) is str else print(check_errortype(email,"str"))

    # изменение количества студентов        
    def set_num_stud(self, students_count) ->  int:
        self.__students_count = students_count if type(students_count) is int else print(check_errortype(students_count,"int"))

    # изменение количества учителей
    def set_num_teachers(self, teachers_count) -> int:
        self.__teachers_count = teachers_count if type(teachers_count) is int else print(check_errortype(teachers_count,"int"))



    # ДОБАВЛЕНИЕ СТУДЕНТА
    def add_student(self,
                    name: str,          # имя
                    familyname: str,    # фамилия
                    age: int,           # возраст
                    gender: str,        # пол
                    nationality: str,   # национальность/гражданство/страна
                    subjects: list      # предметы
                   ):
        
        # создание и добавление объекта 'Student' в список объектов
        # объект ученик
        student = Student(name=name,               # имя
                          familyname=familyname,   # фамилия
                          age=age,                 # возраст
                          gender=gender,           # пол
                          nationality=nationality, # национальность
                          school=self.__name,      # название школы
                          subjects=subjects        # предметы
                         )
        

      
        # данные школ на уровне класса
        self.stud_school_list.append({str(student.get_id): student.get_info})  # добавление ученика в словарь учеников по школе
 

        # общий словарь по всей школе
        self.common_school_info_dict["info"].update([("totstud", str(len(self.stud_school_list)))])   # инфо о школе
        self.common_school_info_dict.update([("students", self.stud_school_list)])       # инфо о студентах школы

        # добавление количества учеников в список информации о школе
        __class__.schools_dict[self.__name]["totstud"] = len(self.stud_school_list)

        print(f"В '{self.__name}' добавлен новый ученик {name} {familyname}, пол '{gender}', возрат {age}, предметы {subjects}")
            

    # ДОБАВЛЕНИЕ УЧИТЕЛЯ
    def add_teacher(self,
                    name: str,          # имя
                    familyname: str,    # фамилия
                    age: int,           # возраст
                    gender: str,        # пол
                    nationality: str,   # национальность/гражданство/страна
                    subjects: list      # предметы
                    ):
        
        # создание и добавление объекта 'Student' в список объектов        
        teacher = Teacher(name=name,               # имя
                         familyname=familyname,   # фамилия
                         age=age,                 # возраст
                         gender=gender,           # пол
                         nationality=nationality, # национальность
                         school=self.__name,      # название школы
                         subjects=subjects        # предметы
                         )      
 
        # данные по школе на уровне класса
        self.teach_school_list.append({str(teacher.get_id): teacher.get_info})  # добавление учителя в словарь учителей по школе

        # общие данные по текущей школе
        self.common_school_info_dict["info"].update([("tottechr", str(len(self.teach_school_list)))]) # количество учителей в школе
        self.common_school_info_dict.update([("teachers", self.teach_school_list)])      # учителя        
        
        # добавление количества учеников в список информации о школе
        __class__.schools_dict[self.__name]["tottechr"] = len(self.teach_school_list)

        print(f"В '{self.__name}' добавлен новый учитель {name} {familyname}, пол '{gender}', возрат {age}, предметы {subjects}")



    @property   # доступ к методу без необходимости прописывать объект внутри скобок
    @staticmethod
    # возвращает словарь с информацией про школу без личной информации студентов / преподавателей
    def get_info(self) -> dict:
        return self.common_school_info_dict["info"]
        

    @property
    @staticmethod
    # ФОРМИРОВАНИЕ ОТЧЁТА ПО ШКОЛЕ
    def get_report(self) -> str:

        # СЛУЖЕБНЫЕ ФУНКЦИИ
        # 
        # чистка от мусорных знаков и пробелов
        def delchars(strow: str) -> str:
            return strow.strip(",") \
                        .replace("'", "") \
                        .replace("  ", " ") \
                        .replace(" , ", ",") \
                        .replace(", '", ",") \
                        .replace(", ", ",") \
                        .replace(" ,", ",") \
                        .replace(", ", ",") \
                        .replace(self.get_name, "") \
                        .replace(",,", ",")
        

        # выборка строк по ученикам и учителям из словаря
        def gethumansrows(humans_dic: dict) -> str:
            list_st1 = humans_dic

            str1 = ""
            # выборка словарей для человека из общего словаря
            for dict1 in list_st1:
                stud_values = dict1.values()
                # выборка значений из конкретного словаря по человеку
                for val in stud_values:
                    stud_values2 = val.values()

                    # формирование строки
                    for key in stud_values2:
                        str1 += str(key) + ","
                # удаление лишних символов
                str1 = delchars(str1) + "\n"  # готовая строка            
            return str1
        

        # формирование имени файла
        filen = self.get_name.replace(" ", "_")
        filen = f"{__class__.mydir}/{filen}.csv"    # путь и имя файла

        # создание каталога
        if not isdir(__class__.mydir): makedirs(__class__.mydir)

        # ЗАПИСЬ В ФАЙЛ
        with open(filen,"w", encoding="UTF-8", newline="") as file:
            # 1/4. ШКОЛА. Заголовок 
            strow = ','.join(head_school_list) + '\n'

            # 2/4. ШКОЛА. Общая запись
            dic0 = self.common_school_info_dict["info"]     # промежуточный словарь с информацией по школе
            strow0 = ','.join(dic0.values()) + '\n'         # формирование строки
            strow += delchars(strow0)                       # удаление лишних символов

            # 3/4. ЛЮДИ. Заголовок 
            strow += ','.join(head_humans_list) + '\n'

            # 4/4. ЛЮДИ. Общая запись
            # ученики
            strow += gethumansrows(self.common_school_info_dict["students"])
            # учителя
            strow += gethumansrows(self.common_school_info_dict["teachers"])

            # запись строки в файл
            file.write(strow)

        print(f"Отчёт по '{self.get_name}' записан в файл '{filen}'")


    @property
    @classmethod
    # информация по всем школам
    def get_total(cls):
        return cls.schools_dict

    @property
    @classmethod
    # количество школ
    def __len__(self):
        return __class__.count
    

# условия запуска модуля при import и самостоятельном
if __name__ == "__main__":
    # ОТОБРАЖЕНИЕ ИНФОРМАЦИИ
    # классы в модуле
    print("Модуль => ", dir())   

    # методы класса
    print(dir(School))
else:
    print("Модуль 'organizations.py' успешно импортирован")
