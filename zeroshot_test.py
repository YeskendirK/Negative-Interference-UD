"""Zero-shot testing:
apply the model directly to the test set without updating on a support batch"""
import logging
import os
import subprocess
import argparse

from udify import util
from allennlp.common import Params
from get_default_params import get_params
from udify.dataset_readers.conll18_ud_eval import evaluate, load_conllu_file, UDError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=9999, type=int, help="Set seed")
    parser.add_argument(
        "--model_dir",
        default=None,
        type=str,
        help="Directory from which to start testing if not starting from pretrain",
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    MODEL_DIR = args.model_dir
    SAVE_DIR = "zeroshot_" + MODEL_DIR.replace("/", "-")
    subprocess.run(["mkdir", SAVE_DIR])

    all_files = list()

    # Load all test set files, including the generated ones
    with open("all_mini_sets.txt", "r") as f:
        for line in f:
            all_files.append(line.strip())

    with open("all_expmix_test.txt", "r") as f:
        for line in f:
            all_files.append(line.strip())

    for test_file in all_files:
        current_gold_file = test_file
        language_name = test_file.split("/")[-1]

        predictions_file = SAVE_DIR + "/" + language_name + "_predicted.conllu"
        performance_file = SAVE_DIR + "/" + language_name + "_performance.json"

        util.predict_and_evaluate_model(
            "udify_predictor",
            get_params("zeroshottesting", args.seed),
            MODEL_DIR,
            current_gold_file,
            predictions_file,
            performance_file,
            batch_size=16,
        )
        print("Wrote", performance_file)


if __name__ == "__main__":
    main()
