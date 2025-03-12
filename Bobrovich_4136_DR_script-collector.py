# Импортируем необходимые библиотеки
import os
import re
import requests

# Шаблоны для извлечения текста из HTML-документа
text_pattern = re.compile("<div id=\"text\".*?>(.*?)<div id=\"tbd\"", re.DOTALL)
span_pattern2 = re.compile("<\/span>([^<].+?)<\/span")
span_pattern = re.compile("<\/span>(.+)<\/span")
strip_pattern = re.compile("&.*?;")
strip_pattern2 = re.compile("<.*?>")

# Шаблон для поиска заголовка книги
title_pattern = re.compile("<div class=title>\n*<h1>(.*?)</h1>", re.DOTALL)

# Шаблон для поиска автора книги
author_pattern = re.compile("<div class=author>(.*?)</div>", re.DOTALL)

# Функция для получения общей информации о книге (заголовок и автор)
def get_info(book, sess):
    r = sess.get(f"https://ilibrary.ru/text/{book}/p.1/index.html")
    author = strip_of_shit(author_pattern.findall(r.text))[0]
    title = strip_of_shit(title_pattern.findall(r.text))[0]
    return (title, author)

# Функция для получения содержимого одной страницы книги
def get_page(book, pnum, sess):
    r = sess.get(f"https://ilibrary.ru/text/{book}/p.{pnum}/index.html")
    text = text_pattern.findall(r.text)[0]
    lines = []
    for t in text.split("\n"):
        res = "".join(span_pattern.findall(t))
        if res != "":
            lines.append(res)
    return strip_of_shit(lines)

# Функция для очистки строк от HTML-сущностей и тегов
def strip_of_shit(lines):
    for i in range(0, len(lines)):
        lines[i] = "".join(strip_pattern.split(lines[i]))
        lines[i] = "".join(strip_pattern2.split(lines[i]))
        lines[i] = lines[i].lstrip()
    return lines

# Основная функция для скачивания всей книги
def get_book(book, session):
    try:
        info = get_info(book, session)
    except Exception as e:
        print(f"Error getting info for book {book}, {e}")
        return
    title = info[0]
    author = info[1]
    
    title2 = (title[:75] + '..') if len(title) > 40 else title
    print(f"Book {book}: {title2} - {author}")
    filestring = "books/"+ author + "/" + title2 + "_" + str(book)
    os.makedirs(os.path.dirname(filestring), exist_ok=True)
    f = open(filestring, "w")
    i = 1
    while True:
        try:
            p = get_page(book, i, session)
        except:
            break
        for s in p:
            f.write(s+"\n")
        f.write("\n")
        i += 1
    f.close()

# Создаем сессию для запросов
session = requests.Session()
# Цикл для обработки книг с номерами от 1 до 4588
for i in range(1,25):
    get_book(i, session)
