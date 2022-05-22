import os
import json
import argparse
from collections import defaultdict

'''
Example: 
python read_meta_test_outputs.py --output_file pretrained_only.json --results_dir metatesting_0.0001_0.0001_20_20_sgd_ONLY 
'''

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output_file", type=str, default='dummy.json', help="The path to output the concatenated files"
)
parser.add_argument(
    "--results_dir",
    default=None,
    type=str,
    help="The path containing all json files with meta test results",
)
args = parser.parse_args()

all_data = defaultdict(list)
for file in os.listdir(args.results_dir):
    if file.endswith('.json'):
        lang_name = file.split('-')[0]
        lang_name = lang_name.split('_')[1]
        with open(os.path.join(args.results_dir, file), 'r') as f:
            file_data = json.load(f)

        all_data[lang_name].append(float(file_data['LAS']['aligned_accuracy']))

mean_data = {}
for lang in all_data.keys():
    mean_data[lang] = sum(all_data[lang]) / len(all_data[lang])

log_dir = "metatest_results_all_lang/"
if os.path.exists(log_dir) is False:
    os.makedirs(log_dir)

with open(log_dir + args.output_file, 'w') as outfile:
    json.dump(mean_data, outfile, indent=4)
    print("Meta test avg. results are saved in json file:", args.output_file)
