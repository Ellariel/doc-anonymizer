import stanza
#stanza.download('ru')

import spacy
# !python3 -m spacy download ru_core_news_sm
# !python3 -m spacy download ru_core_news_lg

import re


def extract_full_name(corpus, lib='stanza'):
    
    corpus = preprocess_text(corpus)
    
    if lib == 'stanza':
        result = stanza_process(corpus)
        
    elif lib == 'spacy':
        result = spacy_process(corpus)
    
    full_name_list = []
    for name in result:
        if re.findall(r'[А-Я]{1}\.[А-Я]*', str(name)) != []: # если регулярка что-то нашла
            full_name_list.append(str(name).strip())
        else:                                     # если регулярка ничего не нашла
            new_name = re.split(' ', str(name))
#             new_name = str(name).split(' ')
            for name_ in new_name:
                full_name_list.append(str(name_).strip()) # возможно, тут не надо стрипить - показатель куска перенесенного слова
                       
    full_name_list = postprocess_text(full_name_list)        
    
    return full_name_list


def preprocess_text(corpus):
    new_c = re.sub('[$|@|&|"||]', '', corpus)
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
    
    
def spacy_process(corpus):
    nlp = spacy.load('ru_core_news_lg')
    doc = nlp(corpus)
    result = [ent for ent in doc.ents if ent.label_ == 'PER']
    return result


def postprocess_text(old_list):
    old_list = list(filter(None, old_list))
    
    new_list = []
    for elem in old_list:
        if len(elem) < 3:
            continue
        elif len(elem) == 3 and re.findall(r'[А-Я]{3}', str(elem)) != []:
            continue
        else:
            new_list.append(elem)
            
    return new_list
