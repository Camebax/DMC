filepath_1 = r"C:\Users\Maxim\Downloads\Cleaning_hall_dmc_3\114 Строительная часть. Буммонтаж 1932г\107 Поперечный.pdf"
filepath_2 = r"C:\Users\Maxim\Downloads\Cleaning_hall_dmc_3\122_1965\238559.pdf"
filepath_3 = "Russian_alphabet.pdf"
import cv2
import numpy as np
import os
import fitz
from PIL import Image
from pytesseract import pytesseract
pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'


# Открываем файлы с картинками
#img_to_read = cv2.imread(filepath_2)
#cv2.imshow('Result',img_to_read)
#cv2.waitKey(0)
#pdf = fitz.open(filepath_2)
#page = pdf.loadPage(0)
#zoom = 10    # zoom factor
#mat = fitz.Matrix(zoom, zoom)
#pix = page.getPixmap(matrix = mat)
#pix = page.getPixmap()
# path_for_img = "C:\Users\Maxim\Downloads\Cleaning_hall_dmc_3\122_1965\238559.pdf" - "pdf" + "png"

'''
def pdf_to_png(filepath):
    doc = fitz.open(filepath)
    zoom = 3    # zoom factor
    page = doc.loadPage(0)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.getPixmap(matrix = mat)
    pix.writePNG(filepath.replace('pdf','png'))
'''
#curImg = cv2.imread('learn_data/letters_а/а_531.PNG')
#curImg = cv2.resize(curImg,(32,32))
#   print(curImg.shape)

# for the_file in os.listdir('learn_data/letters_а'):
#     print(the_file)
#     curImg = cv2.imread('learn_data/letters_а/'+the_file)
#     curImg = cv2.resize(curImg,(32,32))
#     print(curImg.shape)

# curImg = cv2.imread('learn_data/letters_a/'+'a_7.png')
# curImg = cv2.imdecode(np.fromfile('learn_data/letters_б/'+'б_103.png', dtype=np.uint8), cv2.IMREAD_UNCHANGED)
# curImg = cv2.resize(curImg,(32,32))
# print(curImg.shape)

print(ord('~'))