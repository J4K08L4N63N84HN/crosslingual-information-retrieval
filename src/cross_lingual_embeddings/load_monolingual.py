import io
import numpy as np
import codecs


def load_embedding(fname, number_tokens=5000):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    words = []
    embedding = []
    for index, line in enumerate(fin):
        if index == 0:
            continue
        tokens = line.rstrip().split(' ')
        words.append(tokens[0].lower())
        embedding.append(tokens[1:])
        if index == number_tokens:
            return words, np.array(embedding).astype("float32")
    return words, np.array(embedding)


def load_translation_dict(dict_path):
    translation_source = []
    translation_target = []
    for line in list(codecs.open(dict_path, "r", encoding='utf8', errors='replace').readlines()):
        line = line.strip().split("\t")
        translation_source.append(line[0].lower())
        translation_target.append(line[1].lower())
    return translation_source, translation_target
