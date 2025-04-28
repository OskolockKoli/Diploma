import os
import random
from shutil import copyfile

def reduce_books_percentage(input_folder, output_folder, percentage):
    # Проверяем, существует ли входная папка
    if not os.path.exists(input_folder):
        print(f"Папка {input_folder} не найдена.")
        return

    # Создаем выходную папку, если она еще не существует
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Проходимся по всем подпапкам в input_folder
    for genre_folder in os.listdir(input_folder):
        # Разбираем название папки жанра
        genre_name = genre_folder.split('+')
        if len(genre_name) == 1:
            genre_name = genre_name[0].split('-')[0]
        else:
            genre_name = '+'.join([name.split('-')[0] for name in genre_name])
        
        count_str = genre_folder.split('-')[-1]
        try:
            count = int(count_str)
        except ValueError:
            continue  # Пропускаем папки, у которых не удается распознать количество произведений
        
        # Вычисляем новое количество произведений после уменьшения на заданный процент
        new_count = int(count * (1 - percentage / 100))
        if new_count <= 0:
            continue  # Если произведений не осталось, пропускаем эту папку
        
        # Формируем новое имя папки жанра
        new_genre_folder_name = f"{genre_name}-{new_count}"
        
        # Копируем папку жанра в новую папку с новым именем
        src_path = os.path.join(input_folder, genre_folder)
        dst_path = os.path.join(output_folder, new_genre_folder_name)
        os.makedirs(dst_path)
        
        # Получаем список всех файлов в исходной папке
        all_files = os.listdir(src_path)
        if len(all_files) > new_count:
            # Удаление лишних файлов
            files_to_remove = random.sample(all_files, len(all_files) - new_count)
            for file_to_remove in files_to_remove:
                os.remove(os.path.join(src_path, file_to_remove))
            
            # Обновляем список оставшихся файлов
            remaining_files = set(all_files) - set(files_to_remove)
        else:
            remaining_files = all_files
        
        # Копируем оставшиеся файлы в новую папку
        for book_file in remaining_files:
            src_book_path = os.path.join(src_path, book_file)
            dst_book_path = os.path.join(dst_path, book_file)
            copyfile(src_book_path, dst_book_path)
    
if __name__ == "__main__":
    input_folder = r'C:\Users\User\AppData\Local\Programs\Python\Python310\books_max_balanced'
    output_folder = 'books_new_balanced'
    percentage = float(input("Введите процент уменьшения общего количества произведений: "))
    reduce_books_percentage(input_folder, output_folder, percentage)
