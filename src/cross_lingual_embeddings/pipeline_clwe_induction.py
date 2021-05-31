import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from load_monolingual import save_clew
from modify_dictionary import cut_dictionary_to_vocabulary
from supervised_cle import Projection_based_clwe
from evaluation_bli import Evaluator
from unsupervised_cle import VecMap
from text_encoders import TextEncoders


def clew_induction(path_source_language, path_target_language, train_translation_dict_path,
                   train_translation_dict_1k_path, test_translation_dict_path, new_test_translation_path,
                   name_translation, number_tokens=100000, save_embedding=False):


    print("\nFirst, we cut the test dictionaries to the monolingual vocabularies:")
    cut_dictionary_to_vocabulary(path_source_language, path_target_language,
                                 test_translation_dict_path, new_test_translation_path,
                                 number_tokens=number_tokens)

    test_translation_dict_path = new_test_translation_path

    # PROC - 5K dictionary
    print("--------------------------------")
    print("\nCreate procrustes model with 5000 translation pairs")
    proc_algorithm = Projection_based_clwe(path_source_language, path_target_language,
                                           train_translation_dict_path, number_tokens=number_tokens)

    proc_algorithm.proc(source_to_target=True)
    Evaluator(proc_algorithm, test_translation_dict_path).evaluation_on_BLI()
    if save_embedding:
        save_clew(proc_algorithm, name_translation + "_proc_5k")
    del proc_algorithm

    # PROC - 1K dictionary
    print("--------------------------------")
    print("\nCreate procrustes model with 1000 translation pairs")
    proc_algorithm = Projection_based_clwe(path_source_language, path_target_language,
                                           train_translation_dict_1k_path, number_tokens=number_tokens)

    proc_algorithm.proc(source_to_target=True)
    Evaluator(proc_algorithm, test_translation_dict_path).evaluation_on_BLI()
    if save_embedding:
        save_clew(proc_algorithm, name_translation + "_proc_1k")
    del proc_algorithm

    # PROC-B - 1K dictionary
    print("--------------------------------")
    print("\nCreate procrustes bootstrapping model with 1000 translation pairs")
    proc_b_algorithm = Projection_based_clwe(path_source_language, path_target_language,
                                             train_translation_dict_1k_path, number_tokens=number_tokens)

    proc_b_algorithm.proc_bootstrapping(growth_rate=1.5, limit=10000)
    Evaluator(proc_b_algorithm, test_translation_dict_path).evaluation_on_BLI()
    if save_embedding:
        save_clew(proc_b_algorithm, name_translation + "_proc_b_1k")
    del proc_b_algorithm

    # Unsupervised VecMap
    print("--------------------------------")
    print("\nCreate VecMap model")
    vec_map = VecMap(path_source_language, path_target_language, number_tokens=100000)
    # Please use GPU if available and install cupy
    use_gpu = True
    vec_map.build_seed_dictionary(use_gpu)
    vec_map.training_loop(use_gpu)
    Evaluator(vec_map, test_translation_dict_path).evaluation_on_BLI()
    if save_embedding:
        save_clew(vec_map, name_translation + "_vecmap")
    del vec_map

    # Text Encoder First Layer
    print("--------------------------------")
    print("\nCreate  Text Encoder First Layer model")
    xlm_r = TextEncoders("xlm-r")
    xlm_r.create_source_target_embedding(test_translation_dict_path, use_layer=1)
    Evaluator(xlm_r, test_translation_dict_path).evaluation_on_BLI()
    del xlm_r

    # Text Encoder Last Layer
    print("--------------------------------")
    print("\nCreate  Text Encoder Last Layer model")
    xlm_r_last_layer = TextEncoders("xlm-r")
    xlm_r_last_layer.create_source_target_embedding(test_translation_dict_path, use_layer=12)
    Evaluator(xlm_r_last_layer, test_translation_dict_path).evaluation_on_BLI()
    del xlm_r_last_layer

