"""
Creates an old-style allennlp "dataloader" before pytorch niceness.
Takes into account the languages without a training set -
 Make sure you have manually created the training sets for each of these.
"""

import argparse
import os
import datetime
import allennlp
from allennlp.common.params import Params
from allennlp.nn.util import move_to_device
from allennlp.common.util import lazy_groups_of
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.util import datasets_from_params
from udify import util
from naming_conventions import train_languages, validation_languages
from naming_conventions import (
    languages_too_small_for_20_batch_20,
    languages_too_small_for_20_batch_20_lowercase,
)
import torch

from torch.utils.data import DataLoader

print(f"[INFO]: Using the following train languages", train_languages)

def get_test_set(language, language2, number=None, validating=False, bs=20):
    """
    Gets the test set - for most languages this is just the normal test file.
    For some low resource languages it is another file, dependent on the number (crossvalidation)
    """
    if language in languages_too_small_for_20_batch_20:
        testpath = os.path.join(
            "data/ud-tiny-treebanks/size{0}/".format(bs),
            language2 + "-test" + str(number) + ".conllu",
        )
    # This is nice since the dev/test for both Telugu and Bulgarian are about the same size
    # We can use the validation set during meta-validation.
    elif language in validation_languages and validating:
        testpath = os.path.join(
            "data/ud-treebanks-v2.3", language, language2 + "-dev.conllu"
        )
    else:
        testpath = os.path.join(
            "data/ud-treebanks-v2.3", language, language2 + "-test.conllu"
        )
    return testpath


def get_language_dataset(
    language,
    language2,
    seed,
    support_set_size=32,
    validate=False,
    number=None,
    cpu_for_some_reason=False,
    ranout=False,
    bs=20,
    get_test_instead_of_validation=False,
    no_tricks=False,
):
    """
    A helper function that returns an Iterator[List[A]]
    Args:
        language: the uppercased variant, ie. UD_Russian-Taiga or Portugese-GSD
        language2: the lowercased variant, ie. ru_taiga-ud or pt_gsd-ud
        Why are these files named this way? I do not know, one of the mysteries of udify
    """
    print(f"[INFO]: Using support set size", support_set_size)
    configs = []
    the_params = {
        "name": "clean_dataload",
        "base_config": "config/udify_base.json",
        #'device':-1, # set for cpu
        "predictor": "udify_predictor",
    }

    serialization_dir = os.path.join(
        "logs",
        the_params["name"],
        datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S"),
    )

    overrides = {}
    # if args.device is not None:
    #    overrides["trainer"] = {"cuda_device": args.device}
    # if args.lazy is not None:
    if cpu_for_some_reason:
        overrides["trainer"] = {"cuda_device": -1}
    overrides["dataset_reader"] = {"lazy": True}

    trainpath = os.path.join(
        "data/ud-treebanks-v2.3", language, language2 + "-train.conllu"
    )
    valpath = os.path.join(
        "data/ud-treebanks-v2.3", language, language2 + "-dev.conllu"
    )
    testpath = os.path.join(
        "data/ud-treebanks-v2.3", language, language2 + "-test.conllu"
    )

    if language in languages_too_small_for_20_batch_20 and not no_tricks:
        trainpath = os.path.join(
            "data/ud-tiny-treebanks/size{0}/".format(bs),
            language2 + "-dev" + str(number) + ".conllu",
        )
        testpath = os.path.join(
            "data/ud-tiny-treebanks/size{0}/".format(bs),
            language2 + "-test" + str(number) + ".conllu",
        )
    elif ranout:
        trainpath = os.path.join(
            "data/ud-treebanks-v2.3", language, language2 + "-dev.conllu"
        )

    iterator_params = {
        "batch_size": support_set_size,
        "maximum_samples_per_batch": [
            "num_tokens",
            support_set_size * 80,
        ],  # Unfortunately necessary
    }

    if validate:
        iterator_params = {
            "batch_size": support_set_size,
        }

    configs.append(Params(overrides))
    configs.append(
        Params(
            {
                "train_data_path": trainpath,
                "validation_data_path": (
                    valpath
                    if os.path.exists(valpath) and language in train_languages
                    else trainpath
                )
                if not no_tricks
                else (testpath if get_test_instead_of_validation else valpath),
                "test_data_path": testpath,
                "vocabulary": {
                    "directory_path": os.path.join(
                        "data/vocab/english_only_expmix4/vocabulary"
                    )
                },
                "iterator": iterator_params,
                "random_seed": seed,
                "numpy_seed": seed,
                "pytorch_seed": seed,
            }
        )
    )

    configs.append(Params.from_file("./config/ud/en/udify_bert_finetune_en_ewt.json"))
    configs.append(Params.from_file(the_params["base_config"]))

    params = util.merge_configs(configs)

    if "vocabulary" in params:
        # Remove this key to make AllenNLP happy
        params["vocabulary"].pop("non_padded_namespaces", None)
    params["device"] = 0
    vocab = util.cache_vocab(params)
    # Special logic to instantiate backward-compatible trainer.
    # print(params)
    datasets = datasets_from_params(params)
    # print(pieces)
    if validate:
        raw_train_generator = datasets["validation"]
    else:
        raw_train_generator = datasets["train"]
    
    # for dataset in raw_train_generator.values():
    
    vocab = Vocabulary.from_files('data/vocab/english_only_expmix4/vocabulary')
    raw_train_generator.index_with(vocab)
    from allennlp.data import allennlp_collate
    # Construct a dataloader directly for a dataset which contains allennlp
    # Instances which have _already_ been indexed.
    # raw_train_generator = move_to_device(raw_train_generator,torch.device('cuda'))
    my_loader = DataLoader(raw_train_generator, batch_size=support_set_size, collate_fn=allennlp_collate,pin_memory=True)
    # groups = lazy_groups_of(my_loader
# , 1)# -R hardcoded batch of size 1?
    
    return iter(my_loader)