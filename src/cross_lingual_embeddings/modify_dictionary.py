import csv
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from load_monolingual import load_translation_dict, load_embedding


def cut_dictionary_to_vocabulary(path_source_language, path_target_language, translation_dict_path, new_translation_dict_path, number_tokens=5000):

    src_embedding_word, _ = load_embedding(path_source_language, number_tokens)
    trg_embedding_word, _ = load_embedding(path_target_language, number_tokens)
    translation_source, translation_target = load_translation_dict(translation_dict_path)
    
    src_word2ind = {word: i for i, word in enumerate(src_embedding_word)}
    trg_word2ind = {word: i for i, word in enumerate(trg_embedding_word)}

    in_voc = []
    out_voc = []

    for index_translation in range(len(translation_source)):
        source_word = translation_source[index_translation]
        target_word = translation_target[index_translation]
        if source_word not in src_word2ind.keys():
            out_voc.append(index_translation)
            continue
        if target_word not in trg_word2ind.keys():
            out_voc.append(index_translation)
            continue

        in_voc.append(index_translation)

    new_translation_dict = [[translation_source[index], translation_target[index]] for index in in_voc]
    print("Original Dictionary Size: {}".format(len(translation_source)))
    print("New Dictionary Size: {}".format(len(new_translation_dict)))
    with open(new_translation_dict_path, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(new_translation_dict)