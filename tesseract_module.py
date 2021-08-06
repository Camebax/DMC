import cv2
import os
import fitz
import shutil
import numpy as np
from PIL import Image
from pytesseract import pytesseract

pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'
config = r'--oem 3 --psm'

# i  в аргументах - номер итерации, чтобы вырезанных символов не пересекались
def create_learn_base(filename, language, i):  # Создает папки с вырезанными распознанными символами в папке learn_data
    # Открываем файлы с картинками
    img_to_read = cv2.imdecode(np.fromfile('photo_files\{}'.format(filename), dtype=np.uint8),cv2.IMREAD_UNCHANGED)  # МОДУЛЬ ДЛЯ ЧТЕНИЯ РУССКИХ ФАЙЛОВ #
    img_to_crop = Image.open('photo_files\{}'.format(filename))

    # Считываем текст с картинки в массив, если нужно - выводим
    # words_in_image = pytesseract.image_to_string(img_to_read, lang=language)
    # print(words_in_image)

    height, width, c = img_to_read.shape
    letter_boxes = pytesseract.image_to_boxes(img_to_read, lang=language)

    for box in letter_boxes.splitlines():  # Вырезаем по очереди квадраты с символами
        # Обрабатываем ошибки, возникающие при выходе за пределы картинки при обрезке
        try:
            i += 1
            box = box.split()
            x, y, w, h = int(box[1]), int(box[2]), int(box[3]), int(box[4])
            cv2.rectangle(img_to_read, (x, height - y), (w, height - h), (0, 0, 255), 1)
            area = (x, height - h, w, height - y)  # Задаем область, содержащую вырезаемый символ
            cropped_img = img_to_crop.crop(area)
            try:  # Обрабатываем ошибки, возникающие при неправильных именах файлов
                if not os.path.exists('learn_data\s_{}'.format(box[0])):
                    os.mkdir('learn_data\s_{}'.format(box[0]))
                cropped_img.save('learn_data\s_{}/{}_{}.PNG'.format(box[0], box[0], i))
            except OSError:
                pass
        except SystemError:
            pass
    return i

# Возвращает путь к картинке, созданной на основе 1 СТРАНИЦЫ pdf файла
# На входе требуется название pdf файла
def pdf_to_png(filename):
    doc = fitz.open('pdf_files\{}'.format(filename))
    zoom = 4  # zoom factor (влияет на качество получаемого из pdf изображения png)
    page = doc.loadPage(0)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.getPixmap(matrix=mat)
    new_filename = filename.replace('pdf', 'png')
    pix.writePNG('photo_files\{}'.format(new_filename))
    return new_filename


def clear_directory(directory):
    shutil.rmtree(directory)
    os.makedirs(directory)


# clear_directory('learn_data')

# for the_file in os.listdir('pdf_files'):
#     #print(the_file)
#     filename = the_file
#     png_filename = pdf_to_png(filename)
#     #print(png_filename)

i = 0
for the_file in os.listdir('photo_files'):
    i += create_learn_base(the_file, 'rus', i)
    # print(i)