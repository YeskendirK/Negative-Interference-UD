# Measuring Negative Interference in Cross-Lingual Adaptation in Dependency Parsing
This repo is the project to measure negative interference in a multilingual meta-learning setup for the task of dependency parsing. 

We built upon the paper [Meta-learning for fast cross-lingual adaptation in dependency parsing](https://arxiv.org/abs/2104.04736) and [On Negative Interference in Multilingual Models:
Findings and A Meta-Learning Treatment]() along the codebase of Udify. 

# Environment Setup
Ideally setup a conda environment and install all the requirements. jobfiles folder consist of all the required .sh files to run on Lisa. 
Use lisaatcs.job to setup your environment.

# Downloading Dataset and Setting up Project
Create the directory for the data in `Negative-Interference-UD`:
```
mkdir -p data/ud-treebanks-v2.3
mkdir -p data/exp-mix
mkdir -p data/concat-exp-mix
```

Navigate back to the `metalearning` directory (`cd ..`) and download the data.
```
bash ./scripts/download_ud_data.sh
```
It seems that `download_ud_data.sh` not only downloads the data but also creates a treebank for all languages.

Run a script that copies treebanks of all languages used in her paper (based on Table 7). You can run it in the root metalearning directory.
```
python scripts/make_expmix_folder.py
```

Afterward, you can just pass the name of the folder with all these treebanks to concatenate them. `concat_treebanks.py` needs imports Udify's `util.py` which imports stuff like torch, so we need to run `concat_treebanks.py` in a batch script. For that, you can use `concat_treebanks.sh`. Run it from the root directory of metalearning with the command:

```
sbatch concat_treebanks.sh
```

After concatenating treebanks of all relevant languages, create the vocabulary (around 15 minutes):
```
sbatch create_vocabs.sh
```

Refer to the config file 'config/ud/en/udify_bert_finetune_en_ewt.json' to change to proper vocabulary path as Udify copies the vocabulary in multiple places through the train and test process.

## Training Pipeline

# Pre-train mBert
We use many pre-training languages. Example job files are present in 'jobfiles/' directory.

As an example to finetune on Hindi run `hindipretrain.job`. Refer to the paper for parameters and do not forget to change the 'path' in the respective config file. 

## Setup meta-learning and cosine similarity calculation

1. Add pytorch and other libs to env if they weren't added before.
2. Check your unique path to the pre-trained mBERT generated from pretraining. Check the 'logs/' folder for generated logs.
3. Fine-tuning process creates a file `model.tar.gz` and other metadata including best.th.(Note: some of the branches might not have this updated, so ensure that the model.tar.gz is zipped in the same location' and rename the `weights.th` into `best.th` with `mv weights.th best.th`) 
4. Modify `train_meta.sh` to use the correct --model_dir from your pretraining. Change the flags as desired. With default parameters, it takes around 20 hours.
5. As an example run `hindimetatrain.sh` for the hindi pre-trained model
6. The numpy array containing gradient similarities is located in `cos_matrices`. The checkpoint gradient similarities are saved every `save_every` parameter.


**NOTE:** It is not possible to run the full training with a GPU with less than **24GB** of memory! So when using Lisa we need to use the RTX titan. The job file already uses this (_gpu_titanrtx_shared_course_). Even with GPU equipped with 24GB memory OOM errors might occur! 

### Evaluation and Meta-testing

To do evaluation or Meta-testing we use the script `metatest_all.py`. It will generate a folder like `metavalidation_0.0001_1e-05_20_20_sgd_saved_models-XMAML_0.001_0.001_0.001_0.001_5_9999_1` with the scores in json files.

### Evaluation

Run `python metatest_all.py --validate True --lr_decoder 0.0001 --lr_bert 1e-04 --updates 20 --support_set_size 20 --optimizer sgd --seed 3 --episode 500 --model_dir saved_models/XMAML_0.0005_5e-05_0.0005_5e-05_20_9999`  where the path for `--model_dir` was created after running `train_meta.py` and the filepath corresponds to the params of the run.  _This can be done without the RTX gpu._

### Meta-testing

For this, we will need the tiny-treebanks split for cross-validation. Run `python split_files_tiny_auto.py` and it will take care of making the test files. 
We run the same command as for validation but without the --validate flag. `python metatest_all.py --lr_decoder 0.0001 --lr_bert 1e-04 --updates 20 --support_set_size 20 --optimizer sgd --seed 3 --episode 500 --model_dir saved_models/XMAML_0.0005_5e-05_0.0005_5e-05_20_9999
`  
_Need more than 8gb of gpu memory._

### Quick Results

You can visualize the gradient conflicts generated in the cos_matrices directory. Use _visualize.ipynb_ to generate conflict graph and epoch level gradient information.
