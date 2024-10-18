# КЛАССЫ
# Этот модуль содержит классы объектов 'Human', 'Student', 'Teacher'

# условия запуска модуля при import и самостоятельном
if __name__ != "__main__":
    # константы
    from src.utils.listutils import (status_student,
                                    status_teacher
                                    )

# ФУНКЦИЯ ВЫВОДА ОЖИДАЕМОГО ТИПА АРГУМЕНТА
# переменная и ожидаемый тип
def check_errortype(arg=None,typearg: str=None) -> None:
    return f"!!!АРГУМЕНТ НЕ СОЗДАН!!!. Аргумент '{arg}' должен быть типа '{typearg}'"

# ЧЕЛОВЕК
class Human:

    # счётчик класса
    count = 0

    def __init__(self,
                name: str,
                familyname: str,
                age: int,
                gender: str,
                nationality: str
                ) -> None:
        
        # проверка на соответствие типов данных родительского класса
        self.__name = name if type(name) is str else print(check_errortype(name,"str"))                              # имя
        self.__familyname = familyname if type(familyname) is str else print(check_errortype(familyname,"str"))      # фамилия
        self.__age = age if type(age) is int else print(check_errortype(age,"int"))                                  # возраст
        self.__gender = gender if type(gender) is str else print(check_errortype(gender,"str"))                      # пол
        self.__nationality = nationality if type(nationality) is str else print(check_errortype(nationality,"str"))  # национальность/гражданство/страна

        self.__id: int = __class__.count   # id объекта класса 'Human'
        __class__.count += 1


    def __str__(self) -> str:
        return f"Это класс {self.__class__.__name__}"


    def __len__(self) -> int:
        __class__.count



    # ДОСТУП К АТРИБУТАМ
    @property
    def get_name(self):
        return self.__name          # имя
    
    @property    
    def get_familyname(self):
        return self.__familyname    # фамилия
    
    @property        
    def get_age(self):
        return self.__age           # возраст
    
    @property        
    def get_gender(self):
        return self.__gender        # пол
    
    @property        
    def get_nationality(self):
        return self.__nationality   # национальность


    # ИЗМЕНЕНИЕ АТРИБУТОВ
    # изменение имени
    def set_name(self, name: str) -> str:
        self.__name = name if type(name) is str else print(check_errortype(name,"str"))

    # изменение фамилии
    def set_family(self, familyname: str) -> str:
        self.__familyname = familyname if type(familyname) is str else print(check_errortype(familyname,"str"))

    # изменения возраста
    def set_age(self, age: int) -> int:
        self.__age = age if type(age) is int else print(check_errortype(age,"int"))

    # изменение пола
    def set_gender(self, gender: int) -> str:
        self.__gender = gender if type(gender) is str else print(check_errortype(gender,"str"))

    # изменение национальность
    def set_nationality(self, nationality: str) -> str:
        self.__nationality = nationality if type(nationality) is str else print(check_errortype(nationality,"str"))



    # инфо об объекте
    @property
    @staticmethod
    def get_info(self):
       # возвращает словарь с личной информацией об объекте
        return {"id": self.__id,
                "name": self.__name,
                "familyname": self.__familyname,
                "age": self.__age,
                "gender": self.__gender,
                "nationality": self.__nationality
                }


        


# СТУДЕНТ
class Student(Human):

    # счётчик
    count = 0
    strings_list = []       # строка для записи инфо по ученику
    students_dict = dict()  # словарь со списком объектов ученик

    def __init__(self,
                 name: str,
                 familyname: str,
                 age: int,
                 gender: str,
                 nationality: str,
                 school: str,
                 subjects: list
                ) -> None:

        # наследование методов класса 'Human'
        super().__init__(name, familyname, age, gender, nationality)

        self.__id: int = __class__.count   # id учителя
        self.__school = school             # учебное заведение
        self.__subjects = subjects         # список предметов

        __class__.count += 1               # счётчик

        # словарь объектов класса 'Student'
        __class__.students_dict[self.__id] = {"id": str(self.__id),
                                              "status": status_student,                                              
                                              "name": name,
                                              "familyname": familyname,
                                              "age": str(age),
                                              "gender": gender,
                                              "nationality": nationality,
                                              "school": self.__school,
                                              "subjects": self.__subjects
                                            }
                
        self.strings_list.append(["ученик",
                                  name,
                                  familyname,
                                  age,
                                  gender,
                                  nationality,
                                  self.__school,
                                  self.__subjects]
                                )

    # ДОСТУП К АТРИБУТАМ
    # название учебного заведения
    @property
    def get_school(self) -> str:
        return self.__school
    
    @property
    # название предметов
    def get_subjects(self) -> list:
        return self.__subjects
    
    @property
    def get_id(self) -> int:
        return self.__id

    @property
    @staticmethod
    # информация об объекте
    def get_info(self):
        return __class__.students_dict[self.get_id]
    
    

    # ИЗМЕНЕНИЕ АТРИБУТОВ        
    # изменение названия учебного завежения
    def set_school(self, school:str) -> str:
        if type(school) is str:
            self.__school = school
    

    # добавление предмета в список предметов
    def add_subject(self, subject:str) -> str:
        if type(subject) is str:
            self.__subject = subject                 # предмет
            self.__subjects.append(self.__subject)   # добавление его в список



# УЧИТЕЛЬ
class Teacher(Human):

    # счётчик
    count = 0
    strings_list = []       # строка для записи инфо по учителю
    teachers_dict = dict()  # словарь со списком объектов учитель
    
    def __init__(self,
                 name: str,
                 familyname: str,
                 age: int,
                 gender: str,
                 nationality: str,
                 school: str,
                 subjects: list
                ) -> None:
        
        # наследование методов класса 'Human'
        super().__init__(name, familyname, age, gender, nationality)

        self.__id: int = __class__.count   # id учителя

        self.__school = school if type(school) is str else print(check_errortype(school,"str"))           # учебное заведение

        self.__subjects = subjects if type(subjects) is list else print(check_errortype(subjects,"list"))  # предметы

        __class__.count += 1

        # словарь объектов класса 'Teacher'
        __class__.teachers_dict[self.__id] = {"id": str(self.__id),
                                              "status": status_teacher,                                              
                                              "name": name,
                                              "familyname": familyname,
                                              "age": str(age),
                                              "gender": gender,
                                              "nationality": nationality,
                                              "school": self.__school,
                                              "subjects": self.__subjects                                       
                                              }


        self.strings_list.append([self.__id,
                                  "учитель",
                                  name,
                                  familyname,
                                  age,
                                  gender,
                                  nationality,
                                  self.__school,
                                  self.__subjects]
                                )


    # ДОСТУП К АТРИБУТАМ
    # название учебного заведения
    @property
    def get_school(self) -> str:
        return self.__school
    
    @property
    # название предметов
    def get_subjects(self) -> list:
        return self.__subjects
    
    @property
    def get_id(self) -> int:
        return self.__id

    @property
    @staticmethod
    # информация об объекте
    def get_info(self) -> dict:    
        return __class__.teachers_dict[self.get_id]
            


    # ИЗМЕНЕНИЕ АТРИБУТОВ
    # изменение названия учебного завежения
    def set_school(self, school:str) -> str:        
        self.__school = school
    

    # добавление предмета в список предметов
    def add_subject(self, subject:str) -> str:
        if type(subject) is str:
            self.__subject = subject                 # предмет
            self.__subjects.append(self.__subject)   # добавление его в список


    # def __len__(self):
        # return super().__len__()




# условия запуска модуля при import и самостоятельном
if __name__ == "__main__":
    # ОТОБРАЖЕНИЕ ИНФОРМАЦИИ
    # классы в модуле
    print("Модуль => ", dir())
    print()

    # методы родительского класса
    print("Human => ", dir(Human))
    print()

    # методы дочерних классов
    print("Student => ", dir(Student))
    print()
    print("Teacher => ", dir(Teacher))
  
else:
    print("Модуль 'users.py' успешно импортирован")
