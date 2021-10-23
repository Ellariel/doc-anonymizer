## doc-anonymizer

Anonymizer - инструмент анонимизации чувстивтельных персональных данных в документах, а именно ФИО, путём их закрашивания цветным прямоугольником.
Работает как со сканами документов в PDF или JPG форматах, так и с распространенными офисными форматами: DOC, DOCX, XLS, XLSX, RTF, TXT.

Точность работы в случае сканов существенно зависит от их качества и принципиальной возможности распознавания на них текста.

Задача анонимизации в общем виде решается следующим образом:
* распознаётся текст документа средствами OCR (в случае текстового документа - он извлекается напрямую)
* в тексте производится поиск персональных данных, в данном случае ФИО
* документ конвертируется в графический формат JPG и на нём закрашиваются блоки, содержащие ФИО


### Установка и использование

```shell
#Клонируем с репозитория
#Импортируем библиотеку
from engine import anonymizer

#Вызываем метод
result = anonymizer('sample.pdf')
```

### Используемые технологии

* Tesseract (Apache 2.0)
* Textract (MIT License)
* Stanza (Apache 2.0)


### Контакты разработчиков


* Анна Тимошенко, [a.timoshenko@data-in.ru]
* Мила Минниханова, [m.minnikhanova@data-in.ru]
* Эльвира Гизатуллина, [e.gizatullina@data-in.ru]
* Максим Веденьков, [m.vedenkov@data-in.ru]
* Данила Валько, [d.valko@data-in.ru]

