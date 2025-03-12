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

# Регулярное выражение для поиска всех специальных символов
special_chars_pattern = re.compile(r'[^\w\s-]')

# Функция для очистки текста книги от специальных символов
def sanitize_text(text):
    sanitized_text = special_chars_pattern.sub('', text)
    sanitized_text = ' '.join(sanitized_text.split())
    return sanitized_text

# Шаблон для поиска заголовка книги
title_pattern = re.compile(r'<div class="title">\s*<h1>(.*?)</h1>', re.DOTALL)

# Шаблон для поиска автора книги
author_pattern = re.compile(r'<div class="author">(.*?)</div>', re.DOTALL)

# Функция для очистки строк от HTML-сущностей и тегов
def strip_of_shit(lines):
    for i in range(0, len(lines)):
        lines[i] = "".join(strip_pattern.split(lines[i]))
        lines[i] = "".join(strip_pattern2.split(lines[i]))
        lines[i] = lines[i].lstrip()
    return lines

# Функция для получения общей информации о книге (заголовок и автор)
def get_info(book, sess):
    r = sess.get(f"https://ilibrary.ru/text/{book}/p.1/index.html")
    try:
        author = strip_of_shit(author_pattern.findall(r.text))[0]
    except IndexError:
        print(f"Не удалось найти автора {book}. Пропускаем...")
    try:
        title = strip_of_shit(title_pattern.findall(r.text))[0]
    except IndexError:
        print(f"Не удалось найти название книги {book}. Пропускаем...")
    return (title, author)

# Функция для получения содержимого одной страницы книги
def get_page(book, pnum, sess):
    r = sess.get(f"https://ilibrary.ru/text/{book}/p.{pnum}/index.html")
    
    # Получение всего текста страницы
    page_text = r.text
    
    # Обрабатываем первый вариант структуры
    match_first_variant = re.findall('<z><o>(.+?)</o>(.+?)</z>', page_text, flags=re.DOTALL)
    if match_first_variant:
        text_lines = []
        for match in match_first_variant:
            text_lines.extend(match[0].splitlines())
            text_lines.extend(match[1].splitlines())
    
    # Обрабатываем второй и третий варианты структуры
    elif '<v>' in page_text:
        text_lines = []
        # Разделяем страницу на строки, содержащие теги <v> и </v>
        v_tags = re.finditer('<v>(.*?)</v>', page_text, flags=re.DOTALL)
        for tag_match in v_tags:
            line = tag_match.group(1)
            # Удаляем любые внутренние теги (<s5>, <s8>, <sc> и т.п.)
            cleaned_line = re.sub(r'</?\w+>', '', line)
            cleaned_line = re.sub(r'<script.*?</script>', '', cleaned_line, flags=re.DOTALL) #!!!
            text_lines.append(cleaned_line.strip())
            
    else:
        raise ValueError("Неизвестная структура страницы.")
    
    # Очищаем строки от HTML-сущностей и лишних символов
    clean_lines = strip_of_shit(text_lines)

    return clean_lines

# Основная функция для скачивания всей книги
def get_book(book, session):
    try:
        info = get_info(book, session)
    except Exception as e:
        print(f"Error getting info for book {book}, {e}")
        return
    title = info[0]
    title = sanitize_text(title)
    author = info[1]
    author = sanitize_text(author)
    
    print(f"Book {book}: {title} - {author}")
    filestring = f"books/{author}/{title}_{book}.txt"
    os.makedirs(os.path.dirname(filestring), exist_ok=True)
    f = open(filestring, "w")
    i = 1
    while True:
        try:
            p = get_page(book, i, session)
        except:
            break
        # Здесь фиксируем запись построчно
        for s in p:
            if isinstance(s, str):
                f.write(s + "\n")
            else:
                for item in s:
                    f.write(item + "\n")
        f.write("\n")
        i += 1
    f.close()

# Создаем сессию для запросов
session = requests.Session()
# Цикл для обработки книг с номерами от 1 до 4588
for i in range(1,4589):
    get_book(i, session)
