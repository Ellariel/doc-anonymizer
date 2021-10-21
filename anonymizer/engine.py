from matplotlib import pyplot as plt
import re

# Наши модули
from .tesseract_utils import get_text_corpus
from .recognition_full_name import extract_full_name
from .tesseract_utils import get_text_corpus, get_jpg_anon



def anonymizer(path_to_img):
    # Получаю объект-картинку, оригинал и копию для аномизации
    img = plt.imread(path_to_img)
    img_original = img.copy()
    
    # Получаю корпус текста + таблицу координат
    text_corpus, coordinates = get_text_corpus(img)
    
    # Получаю список ФИО
    full_name_list = extract_full_name(text_corpus)
    
    # Получаю таблицу координат NER-объектов
    full_name_coordinates = get_full_name_coordinates(full_name_list, coordinates)
    
    # Получаю анонимизированную картинку
    jpg_anon = get_jpg_anon(img, full_name_coordinates)
    
    # Показать результат
    show_result(img_original, jpg_anon)


def coordinates_preprocessor(coordinates):
    '''
        Убирает запятые.
        Плюс добавим сюда любую другую предобработку таблицы координат.
    '''
    coordinates['text'] = coordinates['text'].apply(lambda x: re.sub(',', '', x))
    return coordinates


def get_full_name_coordinates(full_name_list, coordinates):
    '''
        Возвращает координаты только найденных ФИО.
    '''
    coordinates_clear = coordinates_preprocessor(coordinates)
    ner_coordinates = coordinates_clear.query(f'text in {full_name_list}')
    return ner_coordinates.reset_index(drop=True)


def show_result(img_original, img_anon):
    '''
        Рисует результат
    '''

    fig, axes = plt.subplots(1, 2)

    # Оригинал картинки
    axes[0].imshow(img_original)
    axes[0].set_title('Оригинальный документ')

    for axis in ['top','bottom','left','right']:
        axes[0].spines[axis].set_linewidth(20)
        axes[0].spines[axis].set_color("whitesmoke")
        axes[0].spines[axis].set_zorder(0)
    axes[0].axes.xaxis.set_visible(False)
    axes[0].axes.yaxis.set_visible(False)

    # Анонимизированная картинки
    axes[1].imshow(img_anon)
    axes[1].set_title('Анонимизированный документ')

    for axis in ['top','bottom','left','right']:
        axes[1].spines[axis].set_linewidth(20)
        axes[1].spines[axis].set_color("whitesmoke")
        axes[1].spines[axis].set_zorder(0)

    axes[1].axes.xaxis.set_visible(False)
    axes[1].axes.yaxis.set_visible(False)

    # Настроить размер
    fig.set_figwidth(19)    #  ширина и
    fig.set_figheight(12)    #  высота "Figure"

    plt.show()
    
        
        
