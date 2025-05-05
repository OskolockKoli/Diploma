import os
import random

def delete_random_txt_files(folder_path, num_to_delete):
    # Получаем список всех .txt файлов в папке
    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    
    # Проверяем, есть ли файлы для удаления
    if len(files) == 0:
        print("Нет .txt файлов в указанной папке.")
        return
    
    # Проверяем, достаточно ли файлов для удаления
    if len(files) <= num_to_delete:
        print(f"Удалено {len(files)} файла(ов), так как больше нет.")
        files_to_delete = files
    else:
        # Выбираем случайные файлы для удаления
        files_to_delete = random.sample(files, num_to_delete)
    
    # Удаление выбранных файлов
    for file_name in files_to_delete:
        file_path = os.path.join(folder_path, file_name)
        try:
            os.remove(file_path)
            print(f"Файл '{file_name}' успешно удалён.")
        except Exception as e:
            print(f"Произошла ошибка при удалении файла '{file_name}': {e}")

# Пример использования
folder_path = r'C:\Users\User\AppData\Local\Programs\Python\Python310\books_1\Статья-2100'  # Укажите путь к вашей папке
num_to_delete = 50 # Количество файлов для удаления

delete_random_txt_files(folder_path, num_to_delete)
