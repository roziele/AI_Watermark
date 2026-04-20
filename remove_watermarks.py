#!/usr/bin/env python3
"""
Скрипт для пакетной обработки изображений: удаление водяных знаков и восстановление текстуры (Inpainting).

Использует алгоритмы OpenCV (Navier-Stokes или Telea) для бесшовного восстановления областей.

Требования:
    - opencv-python
    - numpy

Автор: AI Assistant
"""

import cv2
import numpy as np
import os
from pathlib import Path


# ==================== НАСТРОЙКИ ====================

# Пути к папкам
INPUT_DIR = Path("./input_images")
OUTPUT_DIR = Path("./output_images")

# Радиус восстановления (в пикселях)
# Чем больше значение, тем большая область вокруг маски будет задействована для восстановления
INPAINT_RADIUS = 3

# Алгоритм inpainting: cv2.INPAINT_TELEA или cv2.INPAINT_NS
# INPAINT_TELEA - метод на основе быстрого марширования квадратов (быстрее)
# INPAINT_NS - метод на основе уравнений Навье-Стокса (качественнее для текстур)
INPAINT_METHOD = cv2.INPAINT_NS

# Расширения файлов для обработки
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

# ==================== СОЗДАНИЕ МАСКИ ====================
"""
ВАЖНО: Для работы скрипта необходима маска водяного знака.

Маска - это черно-белое изображение того же размера, что и оригинал, где:
    - Белый цвет (255) обозначает область водяного знака (что нужно удалить)
    - Черный цвет (0) обозначает чистую область (что нужно сохранить)

СПОСОБ 1: Автоматическая генерация маски для водяного знака в фиксированном месте
---------------------------------------------------------------------------
Если водяной знак находится в одном и том же месте на всех изображениях,
укажите координаты прямоугольника в функции create_fixed_mask().

Пример: watermark_rect = (x, y, width, height)
        watermark_rect = (100, 50, 200, 80)  # x=100, y=50, ширина=200, высота=80

СПОСОБ 2: Ручное создание маски в графическом редакторе
---------------------------------------------------------------------------
1. Откройте любое изображение из папки input_images в Photoshop, GIMP или аналоге
2. Создайте новый слой и закрасьте белым цветом область водяного знака
3. Сохраните как отдельный файл с суффиксом "_mask" (например: image001_mask.png)
4. Поместите маску в ту же папку, что и исходное изображение

СПОСОБ 3: Автоматическое обнаружение по цвету (для однотонных водяных знаков)
---------------------------------------------------------------------------
Если водяной знак имеет уникальный цвет, можно использовать цветовую сегментацию.
См. функцию create_color_based_mask() ниже.
"""


def create_fixed_mask(image_shape, watermark_rect):
    """
    Создает маску для водяного знака в фиксированном месте.
    
    Параметры:
        image_shape: кортеж (height, width) изображения
        watermark_rect: кортеж (x, y, width, height) - координаты водяного знака
    
    Возвращает:
        numpy.ndarray: маска размером с изображение (uint8, 0 или 255)
    """
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    x, y, w, h = watermark_rect
    
    # Ограничиваем координаты размерами изображения
    x = max(0, min(x, width))
    y = max(0, min(y, height))
    w = min(w, width - x)
    h = min(h, height - y)
    
    # Закрашиваем область водяного знака белым цветом
    mask[y:y+h, x:x+w] = 255
    
    return mask


def create_color_based_mask(image, lower_color, upper_color):
    """
    Создает маску на основе цветового диапазона водяного знака.
    
    Параметры:
        image: исходное изображение (BGR)
        lower_color: нижняя граница цвета в HSV [H, S, V]
        upper_color: верхняя граница цвета в HSV [H, S, V]
    
    Возвращает:
        numpy.ndarray: бинарная маска
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array(lower_color, dtype=np.uint8)
    upper = np.array(upper_color, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    
    # Морфологические операции для улучшения маски
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.erode(mask, kernel, iterations=1)
    
    return mask


def load_or_create_mask(image_path, image_shape):
    """
    Загружает существующую маску или создает новую.
    
    Порядок поиска маски:
    1. Файл с суффиксом "_mask" (например: image_mask.png)
    2. Файл с префиксом "mask_" (например: mask_image.png)
    3. Автоматическая генерация по фиксированным координатам
    
    Параметры:
        image_path: путь к исходному изображению
        image_shape: размеры изображения (height, width)
    
    Возвращает:
        numpy.ndarray: маска или None если не найдена
    """
    image_path = Path(image_path)
    stem = image_path.stem
    parent = image_path.parent
    
    # Вариант 1: ищем файл вида "image_name_mask.ext"
    for ext in ['.png', '.jpg', '.jpeg']:
        mask_path = parent / f"{stem}_mask{ext}"
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                print(f"  → Загружена маска: {mask_path.name}")
                return cv2.resize(mask, (image_shape[1], image_shape[0]))
    
    # Вариант 2: ищем файл вида "mask_image_name.ext"
    for ext in ['.png', '.jpg', '.jpeg']:
        mask_path = parent / f"mask_{stem}{ext}"
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                print(f"  → Загружена маска: {mask_path.name}")
                return cv2.resize(mask, (image_shape[1], image_shape[0]))
    
    # Вариант 3: используем фиксированные координаты (НАСТРОЙТЕ ПОД СВОЙ ВОДЯНОЙ ЗНАК!)
    # Укажите реальные координаты вашего водяного знака здесь:
    watermark_rect = None  # Пример: (100, 50, 200, 80)
    
    if watermark_rect is not None:
        print(f"  → Создана маска по фиксированным координатам: {watermark_rect}")
        return create_fixed_mask(image_shape, watermark_rect)
    
    # Если ничего не найдено
    return None


def process_image(image_path, output_path, inpaint_radius=INPAINT_RADIUS, method=INPAINT_METHOD):
    """
    Обрабатывает одно изображение: удаляет водяной знак с помощью inpainting.
    
    Параметры:
        image_path: путь к исходному изображению
        output_path: путь для сохранения результата
        inpaint_radius: радиус восстановления в пикселях
        method: алгоритм inpainting (cv2.INPAINT_TELEA или cv2.INPAINT_NS)
    
    Возвращает:
        bool: True если успешно, False если ошибка
    """
    try:
        # Чтение изображения
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"  ✗ Не удалось прочитать изображение: {image_path}")
            return False
        
        # Поиск или создание маски
        mask = load_or_create_mask(image_path, image.shape)
        
        if mask is None:
            print(f"  ⚠ Маска не найдена для: {image_path.name}")
            print(f"    Создайте файл маски с именем '{Path(image_path).stem}_mask.png'")
            print(f"    или укажите координаты водяного знака в переменной 'watermark_rect'")
            return False
        
        # Проверка размеров маски
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        
        # Применение inpainting
        result = cv2.inpaint(image, mask, inpaint_radius, method)
        
        # Создание выходной директории если не существует
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Сохранение результата
        cv2.imwrite(str(output_path), result)
        
        return True
        
    except Exception as e:
        print(f"  ✗ Ошибка при обработке {image_path}: {str(e)}")
        return False


def batch_process_images(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR):
    """
    Пакетная обработка всех изображений в папке.
    
    Параметры:
        input_dir: папка с исходными изображениями
        output_dir: папка для результатов
    """
    # Преобразование путей в объекты Path
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Проверка существования входной директории
    if not input_dir.exists():
        print(f"Ошибка: Папка '{input_dir}' не существует.")
        print("Создайте папку и поместите в неё изображения для обработки.")
        return
    
    # Создание выходной директории
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Поиск всех поддерживаемых изображений
    image_files = []
    for ext in SUPPORTED_EXTENSIONS:
        image_files.extend(input_dir.glob(f"*{ext}"))
        image_files.extend(input_dir.glob(f"*{ext.upper()}"))
    
    # Удаление дубликатов и сортировка
    image_files = sorted(set(image_files))
    
    total = len(image_files)
    
    if total == 0:
        print(f"В папке '{input_dir}' не найдено изображений ({', '.join(SUPPORTED_EXTENSIONS)})")
        return
    
    print(f"Найдено изображений: {total}")
    print(f"Алгоритм: {'Telea' if INPAINT_METHOD == cv2.INPAINT_TELEA else 'Navier-Stokes'}")
    print(f"Радиус восстановления: {INPAINT_RADIUS} px")
    print("-" * 50)
    
    # Счетчики
    success_count = 0
    error_count = 0
    skipped_count = 0
    
    # Обработка каждого изображения
    for i, image_path in enumerate(image_files, 1):
        # Формирование пути для сохранения
        output_path = output_dir / image_path.name
        
        print(f"[{i}/{total}] Обработка: {image_path.name}...", end=" ")
        
        try:
            result = process_image(image_path, output_path)
            
            if result:
                success_count += 1
                print("✓ Готово")
            else:
                skipped_count += 1
                print("⊘ Пропущено")
                
        except Exception as e:
            error_count += 1
            print(f"✗ Ошибка: {str(e)}")
    
    # Итоговая статистика
    print("-" * 50)
    print(f"Обработано: {success_count}/{total}")
    print(f"Пропущено: {skipped_count}/{total}")
    print(f"Ошибок: {error_count}/{total}")
    print(f"Результаты сохранены в: {output_dir.absolute()}")


if __name__ == "__main__":
    print("=" * 50)
    print("Пакетная обработка изображений: удаление водяных знаков")
    print("=" * 50)
    print()
    
    batch_process_images()
    
    print()
    print("Готово!")
