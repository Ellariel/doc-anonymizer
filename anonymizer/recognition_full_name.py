import stanza
stanza.download('ru')

import re
import json
import os
import pathlib


current_directory = str(pathlib.Path(__file__).parent.resolve())
with open(pathlib.Path(f'{current_directory}/ner_exceptions.json'), 'r') as file:
    exceptions_dict = json.load(file)

    
def extract_full_name(corpus):
    
    corpus = preprocess_text(corpus)
    
    result = stanza_process(corpus)
                
    evaluation = get_evaluation(result)   # 0 = уверенность, 1 = могут быть ошибки
    
    full_name_list = []
    for name in result:
        if re.findall(r'[А-Я]{1}\.{1}\s*[А-Я]{1}\.{1}\s*[А-Я]{1}[А-Я|а-я]+', str(name)) != []: # регулярка для поиска имен с инициалами
            
            if filter_result(name, 'initials'):
                full_name_list.append(str(name).strip())    
            
        else:                                     # если регулярка ничего не нашла -> полное фио
            if filter_result(name, 'full'):
                new_name = re.split(' ', str(name))

                for name_ in new_name:
                    full_name_list.append(str(name_).strip()) # возможно, тут не надо стрипить - показатель куска перенесенного слова
                    
    full_name_list = postprocess_text(full_name_list)        
    
    return full_name_list, evaluation


def preprocess_text(corpus):
    new_c = re.sub('[$|@|&|"||]', '', corpus)
    new_c = new_c.replace('\n\n', ', ')
    new_c = new_c.replace('\n', ' ')
    new_c = new_c.replace('  ', ' ')
    
    corpus_list = new_c.split(' ')
    
    new_corpus_list = []
    for n, elem in enumerate(corpus_list):
        
        if len(elem) >= 2 and elem[-1] == '-' and re.findall(r'[А-Я]{1}', corpus_list[n+1]) == []:
            elem = elem.replace('-', '') + corpus_list[n+1]
            corpus_list.pop(n+1)
        new_corpus_list.append(elem)
    
    corpus_str = ' '.join(new_corpus_list)
    corpus_str = re.sub(r'[\d+]', '', corpus_str)
    
    return corpus_str.strip()
    

def stanza_process(corpus):
    nlp = stanza.Pipeline(lang='ru', processors='tokenize,ner', verbose=False) # этих двух процессоров должно быть достаточно под задачу поиска имен
    doc = nlp(corpus)
    result = [ent.text for ent in doc.ents if ent.type == 'PER']
    return result
    

def get_evaluation(result_list):
    
    reg_init = r'[А-Я]{1}\.{1}\s*[А-Я]{1}\.{1}\s*'                                 # регулярка для инициалов
    reg_full = r'[А-Я]{1}[а-я|А-Я]+\s{1}[А-Я]{1}[а-я|А-Я]+\s{1}[А-Я]{1}[а-я|А-Я]+' # регулярка для полных ФИО
    
    evaluation_mark_list = []
    for entity in result_list:
        if re.findall(reg_full, entity) != [] or re.findall(reg_init, entity) != []:
            evaluation_mark_list.append(0)
        else:
            evaluation_mark_list.append(1)
    evaluation_set = set(evaluation_mark_list)
    
    if set(evaluation_mark_list) == {0}:
        evaluation_mark = 0
    else:
        evaluation_mark = 1
            
    return evaluation_mark


def filter_result(entity, name_type):
    
    if name_type == 'initials':
        tmp_entity = entity.lower().strip()
        tmp_entity = tmp_entity.replace(' ', '')
        if tmp_entity in exceptions_dict['initials']:
            return False
        else:
            return True
       
    elif name_type == 'full':
        tmp_entity = entity.lower().strip()
        if tmp_entity in exceptions_dict['full']:
            return False
        else:
            return True
        
    elif name_type == 'other':
        tmp_entity = entity.lower().strip()
        if tmp_entity in exceptions_dict['other']:
            return False
        else:
            return True    


def postprocess_text(old_list):
    old_list = list(filter(None, old_list))
    
    tmp_list = []
    for elem in old_list:
        if len(elem) < 3:
            continue
        elif len(elem) == 3 and re.findall(r'[А-Я]{3}', str(elem)) != []:
            continue
        else:
            tmp_list.append(elem)
    
    new_list = []
    for elem in tmp_list:
        if filter_result(elem, 'other'):
            new_list.append(elem)
    
    return new_list
