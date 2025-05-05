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

# Функция для извлечения ссылок на авторов
def extract_author_links(html):
    soup = BeautifulSoup(html, 'html.parser')
    links = []
    count = 0
    for link in soup.find_all('a'):
        href = link.get('href')
        if href is not None and href.startswith('http://az.lib.ru/'):
            if count == 0:
                count = count + 1
            else:
                author_link = f'{href}'
                #print(author_link)
                links.append(author_link)
    return links

# Функция для извлечения ссылок на произведения конкретного автора
def extract_work_links(author_link, author_html):
    soup = BeautifulSoup(author_html, 'html.parser')
    works = []
    for work_link in soup.find_all('a', href=True):
        href = work_link.get('href')
        if href.endswith('.shtml') and href.startswith('text'):
            work_url = f'{author_link}{href}'
            #print(work_url)
            works.append(work_url)
    return works

# Функция для сохранения текста произведения в файл
def save_text_to_file(text, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(text)

# Функция для извлечения жанра произведения
def extract_genre(soup):
    # Находим первую ссылку, содержащую путь, начинающийся с '/type/index_type_...'
    genre_link = soup.find('a', href=lambda x: x and x.startswith('/type/index_type_'))
    if genre_link:
        genre = genre_link.get_text(strip=True)
        return genre
    return None

# Функция для извлечения названия произведения
def extract_title(soup):
    # Находим тег <title> и извлекаем текст после второй точки
    title_tag = soup.find('title')
    if title_tag:
        title_text = title_tag.text.strip()
        parts = title_text.split(':', maxsplit=1)
        if len(parts) > 1:
            return parts[1].strip()
    return None

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
def parse_authors_and_works():
    # URL главной страницы с авторами
    base_url = 'https://lib.ru/LITRA/'
    
    # Проверяем доступность сервера перед началом парсинга
    if not check_server_availability(base_url):
        print("Сервер недоступен. Парсинг невозможен.")
        return
    
    # Получаем содержимое главной страницы
    main_page_html = get_page_content(base_url)
    
    # Извлекаем ссылки на авторов
    author_links = extract_author_links(main_page_html)
    
    # Создаем общую папку для всех книг
    os.makedirs('books', exist_ok=True)
    
    j = 1
    # Проходимся по каждому автору
    for i, author_link in enumerate(author_links):
        author_name = author_link.split('/')[-1].replace('_', ' ')
        
        # Получаем страницу автора
        author_html = get_page_content(author_link)
        
        # Извлекаем ссылки на произведения автора
        work_links = extract_work_links(author_link, author_html)

        # Проходимся по каждому произведению
        for work_link in work_links:
            while True:
                # Проверяем доступность сервера перед каждым произведением
                if check_server_availability(base_url):
                    break
                else:
                    print("Сервер временно недоступен. Жду 10 секунд...")
                    time.sleep(10)

            # Получаем содержание произведения
            work_html = get_page_content(work_link)

            # Проверяем, что страница загружена успешно
            if work_html is None:
                print(f"Произведение не найдено: {work_link}")
                continue  # Переходим к следующему произведению
            
            # Парсим название произведения
            soup = BeautifulSoup(work_html, 'html.parser')
            title = extract_title(soup)
            if not title:
                title = ""
            cleaned_title = clean_filename(title)
            
            # Извлекаем жанр произведения
            genre = extract_genre(soup)

            # Если жанр не найден, ставим 'Без жанра'
            if not genre:
                genre = 'Без жанра'
            
            # Извлекаем текст произведения
            text_div = soup.find('dd')
            if text_div is not None:
                text = text_div.get_text().strip()
            else:
                text = ""  # Если тег <dd> не найден, присваиваем пустую строку
            
            # Формируем имя файла
            filename = f"books/{genre}_{cleaned_title}_{j}.txt"
            j = j + 1
            
            # Сохраняем текст в файл
            save_text_to_file(text, filename)
            print(f"Сохранено произведение: {filename}")

if __name__ == "__main__":
    parse_authors_and_works()
