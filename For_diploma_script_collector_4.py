import requests
import re
from bs4 import BeautifulSoup
import os
import time

def clean_filename(text):
    # Убираем все недопустимые символы
    invalid_chars = r'[\\/*:?<>|"]'
    cleaned_text = re.sub(invalid_chars, '', text)
    if len(cleaned_text) <= 220:
        return cleaned_text
    else:
        return 'Без названия'

# Функция для получения содержимого страницы
def get_page_content(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        print(f"Ошибка загрузки страницы {url}. Код ошибки: {response.status_code}")
        return None

# Функция для получения авторов и извлечения ссылок их произведения
def extract_authors_and_works(page_html):
    soup = BeautifulSoup(page_html, 'html.parser')
    authors = []
    
    dl_tags = soup.find_all('dl')
    for dl in dl_tags:
        dt_tag = dl.find('dt')
        if dt_tag:
            # Извлечение ФИО автора
            font_tag = dt_tag.find('font', attrs={'color': '#555555'})
            if font_tag:
                author_name = font_tag.text.strip()
                #print(author_name)
            else:
                author_name = ''
            
            # Извлечение ссылки на произведение и его названия
            first_a_tags = dt_tag.find_all('a')
            if first_a_tags:
                work_link = first_a_tags[1]['href']  # Вторая ссылка <a>
                #print(work_link)
                b_tag = first_a_tags[1].find('b')  # Поиск тега <b> внутри второй ссылки <a>
                if b_tag:
                    work_title = b_tag.text.strip()  # Извлечение текста из тега <b>
                    #print(work_title)
                else:
                    work_title = ''  # Если тег <b> не найден
            else:
                work_link = ''
                work_title = ''
            
            # Добавление данных в список
            if author_name and work_link and work_title:
                authors.append((author_name, work_link, work_title))
    
    return authors

# Функция для извлечения текста произведения
def extract_prose_text(prose_html):
    soup = BeautifulSoup(prose_html, 'html.parser')
    text_div = soup.find('dd')
    if text_div is not None:
        text = text_div.get_text().strip()
    else:
        text = ""  # Если тег <dd> не найден, присваиваем пустую строку
    return text

# Функция для сохранения текста произведения в файл
def save_prose_to_file(text, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(text)

# Проверка соединения с сервером
def check_server_availability(url):
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return True
        else:
            return False
    except requests.exceptions.RequestException:
        return False

# Основная функция парсинга
def parse_pages():
    # URL главной страницы с авторами
    base_url = 'http://az.lib.ru/type/index_type_9-{num}.shtml'
    # Создаем папку для хранения файлов
    os.makedirs('Пьеса', exist_ok=True)
    
    # Проходимся по каждой странице
    file_count = 21000  # Начальный порядковый номер
    for page_num in range(1, 12):
        current_base_url = base_url.format(num=page_num)
        print(f"Парсим страницу: {current_base_url}")
        # Проверяем доступность сервера перед началом парсинга
        if not check_server_availability(current_base_url):
            print("Сервер недоступен. Парсинг невозможен.")
            return
        
        # Получаем содержимое текущей страницы
        main_page_html = get_page_content(current_base_url)
        
        # Извлекаем ссылки на авторов и их произведения
        authors = extract_authors_and_works(main_page_html)
        
        # Проходимся по каждому автору и его произведению
        for author_name, work_link, work_title in authors:
            # Получаем текст произведения
            full_work_link = f"http://az.lib.ru/{work_link}"
            work_html = get_page_content(full_work_link)
            prose_text = extract_prose_text(work_html)

            cleaned_author_name = clean_filename(author_name)
            cleaned_work_title = clean_filename(work_title)
            
            # Формируем имя файла
            filename = f"Пьеса/Пьеса_{cleaned_author_name}_{cleaned_work_title}_{file_count}.txt"
            file_count += 1
            
            # Сохраняем текст произведения в файл
            save_prose_to_file(prose_text, filename)
            print(f"Сохранено произведение: {filename}")

if __name__ == "__main__":
    parse_pages()
