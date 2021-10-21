#https://colab.research.google.com/drive/1AOkk-TSDFbvCb3Mxb7sTqRMAFbMi6Ra-?usp=sharing#scrollTo=vgZemXj6hejq
#!apt install tesseract-ocr-rus
#!apt install libtesseract-dev
#!pip install pytesseract
import pytesseract
import cv2
import pandas as pd


def tesseract_enabled():
    return 'rus' in pytesseract.get_languages()


def get_text_corpus(jpg, config=r'-l rus --oem 3 --psm 6'):
    '''
    # Функция принимает на вход jpg, возвращает корпус текста строкой
    # txt, coord = get_text_corpus(img)
    '''
    if not tesseract_enabled():
        raise Exception('Russian is not installed..')
        
    data = pytesseract.image_to_data(jpg, output_type='data.frame', config=config)
    data = data[~pd.isna(data.text)]
    #data.text = data.text.apply(lambda x: x.lower())
    
    if len(data) < 1:
        raise Exception('No words..')
    
    return data.text.str.cat(sep=' '), data #[['left', 'top', 'width', 'height']]


def get_jpg_anon(jpg, coordinates, full_name_list=None, filled=True):
    '''
    # Функция принимает картинку, список имен, возвращает картинку с закрашенными
    # plt.imshow(get_jpg_anon(img, coord))
    '''
    if filled:
        filled = -1
    else:
        filled = 2
    for item in coordinates.iterrows():
        c = item[1]
        jpg = cv2.rectangle(jpg, (c.left, c.top), (c.left + c.width, c.top + c.height), (255, 0, 0), filled)
    return jpg


def get_test(x):
    print(x)
    
    
