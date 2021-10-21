#Для ковертера
#!!add-apt-repository ppa:libreoffice/ppa && sudo apt-get update
#!!apt-get install unoconv
#Для анонимайзера
#!!pip install --upgrade --force-reinstall fitz
#!!pip install --upgrade --force-reinstall pymupdf
#!!pip install webcolors
#Для конвертера pdf2jpg
#!apt-get install poppler-utils
#!pip install pdf2image
import os
import fitz
from webcolors import name_to_rgb

import subprocess
import sys
import re

from pdf2image import convert_from_path
import uuid
import os


def convert_to_pdf(in_file, path=''): #помещает файл с тем же именем, но новым расширением pdf в папку
        args = ['libreoffice', '--headless', '--convert-to', 'pdf', '--outdir', path, in_file]
        process = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=None)
        #print(process.stdout.decode())
        filename = re.search('-> (.pdf) using filter', process.stdout.decode())
        return filename is None


def convert_to_jpg(in_file, out_path='', dpi=300): #помещает файлы картинок в папку и возвращает их список (каждая страница pdf в отдельном файле)
        images = convert_from_path(in_file, dpi=dpi)#, output_folder=path, dpi=300, output_file=str(uuid.uuid4()), use_pdftocairo=True)
        #print(len(images))
        files = []
        for i in range(len(images)):
            filename = page = os.path.join(out_path, str(uuid.uuid4())+'.jpg')
            images[i].save(filename, 'JPEG')
            files.append(filename)
        return files

def _convert_to_jpg(in_file, dpi=300):
        return convert_from_path(in_file, dpi=dpi)
  

def anonymize_pdf(in_file, out_file, text, color='green', filled=True): #создаёт новый файл pdf
        pix = fitz.Pixmap(fitz.csRGB, (0, 0, 300, 300), 0) #просто создается картинка, которая затем помещается на текст
        pix.set_rect(pix.irect, name_to_rgb(color)) #поэтому лучше это вынестив инициализацию, чтобы не плодить сущности
        doc = fitz.open(in_file)
        for page in doc:
            text_instances = page.search_for(text) #ищет все совпадение и для каждого использует маркер
            for inst in text_instances:
                highlight = page.add_highlight_annot(inst)
                page.insert_image(highlight.rect, pixmap=pix, keep_proportion=False, overlay=filled) #помещает на каждый мркер картинку
        doc.save(out_file, garbage=4, deflate=True, clean=True)
        return os.path.exists(out_file)
