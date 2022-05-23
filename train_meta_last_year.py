# -*- coding: utf-8 -*-
"""
This file Meta-Trains on 7 languages
And validates on Bulgarian
"""
import naming_conventions
from pathlib import Path
from get_language_dataset import get_language_dataset
from get_default_params import get_params
from ourmaml import MAML, maml_update
from allennlp.common.util import prepare_environment
from allennlp.models.model import Model
from allennlp.models.archival import archive_model
from schedulers import get_cosine_schedule_with_warmup
from torch import autograd
from torch.optim import Adam
import torch
import numpy as np
import argparse
import subprocess
import sys
import os, glob


from allennlp.nn.util import move_to_device
from sklearn.metrics.pairwise import cosine_similarity

sys.stdout.reconfigure(encoding="utf-8")

torch.cuda.empty_cache()  # please stop oom's


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--skip_update", default=0, type=float, help="Skip update on the support set")
    parser.add_argument("--seed", default=9999, type=int, help="Set seed")
    parser.add_argument("--support_set_size", default=1, type=int, help="Support set size")
    parser.add_argument("--maml", default=False, type=bool, help="Do MAML instead of XMAML, that is, include English as an auxiliary task if flag is set and start from scratch")
    parser.add_argument("--addenglish", default=False, type=bool, help="Add English as a task")
    parser.add_argument("--notaddhindi", default=False, type=bool, help="Add English as a task") #?
    parser.add_argument("--episodes", default=1, type=int, help="Amount of episodes")
    parser.add_argument("--updates", default=5, type=int, help="Amount of inner loop updates")
    parser.add_argument("--name", default="", type=str, help="Name to add")
    parser.add_argument("--meta_lr_decoder", default=0.001, type=float, help="Meta adaptation LR for the decoder")
    parser.add_argument("--meta_lr_bert", default=0.001, type=float, help="Meta adaptation LR for BERT")
    parser.add_argument("--inner_lr_decoder", default=0.001, type=float, help="Inner learner LR for the decoder")
    parser.add_argument("--inner_lr_bert", default=0.001, type=float, help="Inner learner LR for BERT")
    parser.add_argument("--model_dir", default=None, type=str, help="Directory from where to start training. Should be a 'clean' model for MAML and a pretrained model for X-MAML.")
    parser.add_argument("--language_order", default=0, type=int, help="The order of languages in the inner loop")
    parser.add_argument("--save_every", default=4, type=int, help="Save the gradient conflicts every save_every episodes ")
    parser.add_argument("--accumulation_mode", default="sum", type=str, help="What gradient accumulation strategy to use", choices=["mean", "sum"])
    parser.add_argument("--pairwise", default=0, type=int, help="Train pairs of language. Cosine similarity values: 1 - high, 2 - low, 0 - not pairwise")
    args = parser.parse_args()
    print(args)
    Path("saved_models").mkdir(parents=True, exist_ok=True)

    print(f"Using accumulation mode {args.accumulation_mode} for gradient accumulation")
    device = torch.device('cuda')

    training_tasks = []
    torch.cuda.empty_cache()

    # 7 languages by default -R
    if args.language_order == 1:
        lan_ = naming_conventions.train_languages_order_1
        lan_lowercase_ = naming_conventions.train_languages_order_1_lowercase

    elif args.language_order == 2:
        lan_ = naming_conventions.train_languages_order_2
        lan_lowercase_ = naming_conventions.train_languages_order_2_lowercase

    else:
        lan_ = naming_conventions.train_languages
        lan_lowercase_ = naming_conventions.train_languages_lowercase

    #Only for the case if we train  in pairs

    #language pairs with high cosine similarity
    if args.pairwise == 1:
        print('\nkorean-hindi pair\n')
        lan_ = naming_conventions.train_languages_pairwise_1
        lan_lowercase_ = naming_conventions.train_languages_pairwise_lowercase_1
    #language pairs with low cosine similarity
    elif  args.pairwise == 2:
        print('\nkorean-arabic pair\n')
        lan_ = naming_conventions.train_languages_pairwise_2
        lan_lowercase_ = naming_conventions.train_languages_pairwise_lowercase_2
    elif  args.pairwise == 3:
        print('\nczech-arabic pair\n')
        lan_ = naming_conventions.train_languages_pairwise_3
        lan_lowercase_ = naming_conventions.train_languages_pairwise_lowercase_3
    elif  args.pairwise == 4:
        print('\nczech-english pair\n')
        lan_ = naming_conventions.train_languages_pairwise_4
        lan_lowercase_ = naming_conventions.train_languages_pairwise_lowercase_4

   
    for lan, lan_l in zip(lan_, lan_lowercase_):
        if not ("Hindi" in lan and args.notaddhindi):
            training_tasks.append(get_language_dataset(lan, lan_l, seed=args.seed, support_set_size=args.support_set_size))
            
    # Setting parameters
    DOING_MAML = args.maml
    if DOING_MAML or args.addenglish:
        training_tasks.append(get_language_dataset("UD_English-EWT", "en_ewt-ud", seed=args.seed, support_set_size=args.support_set_size))

    UPDATES = args.updates
    EPISODES = args.episodes
    INNER_LR_DECODER = args.inner_lr_decoder
    INNER_LR_BERT = args.inner_lr_bert
    META_LR_DECODER = args.meta_lr_decoder
    META_LR_BERT = args.meta_lr_bert
    SKIP_UPDATE = args.skip_update
    PRETRAIN_LAN = args.model_dir.split("/")[-2] # says what language we are using

    # Filenames
    MODEL_FILE = (args.model_dir if args.model_dir is not None else "logs/bert_finetune_en/2021.05.13_01.56.30")
    maml_string = "saved_models/MAML" if DOING_MAML else "saved_models/XMAML"
    param_list = [
        str(z) for z in [
            maml_string,
            INNER_LR_DECODER,
            INNER_LR_BERT,
            META_LR_DECODER,
            META_LR_BERT,
            UPDATES,
            PRETRAIN_LAN,
            args.seed,
            args.language_order,
            args.accumulation_mode
        ]
    ]
    MODEL_SAVE_NAME = "_".join(param_list)
    MODEL_VAL_DIR = MODEL_SAVE_NAME + args.name
    META_WRITER = MODEL_VAL_DIR + "/meta_results.txt"

    if not os.path.exists(MODEL_VAL_DIR):
        subprocess.run(["mkdir", MODEL_VAL_DIR])
        subprocess.run(["mkdir", MODEL_VAL_DIR + "/performance"])
        subprocess.run(["mkdir", MODEL_VAL_DIR + "/predictions"])
        subprocess.run(["cp", "-r", MODEL_FILE + "/vocabulary", MODEL_VAL_DIR])
        subprocess.run(["cp", MODEL_FILE + "/config.json", MODEL_VAL_DIR])

    with open(META_WRITER, "w") as f:
        f.write("Model ready\n")

    # Loading the model
    train_params = get_params("metalearning", args.seed)
    prepare_environment(train_params)
    m = Model.load(train_params, MODEL_FILE,)
    meta_m = MAML(m, INNER_LR_DECODER, INNER_LR_BERT, first_order=True, allow_unused=True).cuda()
    optimizer = Adam(
        [
            {"params": meta_m.module.text_field_embedder.parameters(), "lr": META_LR_BERT},
            {"params": meta_m.module.decoders.parameters(), "lr": META_LR_DECODER},
            {"params": meta_m.module.scalar_mix.parameters(), "lr": META_LR_DECODER},
        ],
        META_LR_DECODER,
    )

    scheduler = get_cosine_schedule_with_warmup(optimizer, 50, 500)

    # NI START
    print(f"[INFO]: Total amount of training tasks is {len(training_tasks)}")
    cos_matrices = []
    # in the end it will store the following info in each dim - num_episodes x grad_len x num_languages
    # NI END

    def restart_iter(task_generator_, args_):
        """ Restart the iter(Dataloader) by creating again the dataset.
        This method is called when we looped through the whole data and want to start from the beginning.
        """       
        # Get the path to datat like: data/ud-treebanks-v2.3/UD_Arabic-PADT/ar_padt-ud-train.conllu
        task_split = task_generator_._dataset._file_path.split('/')
        language = task_split[-2]  # Language in capitals
        # Language in lower case, remove -train.collu, -dev.conllu, -test.conllu
        language_lowercase_ = task_split[-1].split('-')[0]+'-ud'
        return get_language_dataset(language, language_lowercase_, seed=args_.seed, support_set_size=args_.support_set_size)

    for iteration in range(EPISODES):

        print(f"[INFO]: Starting episode {iteration}\n", flush=True)

        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r - a  # free inside reserved

        print(f'\ntotal     : {t}')
        print(f'reserved  : {r}')
        print(f'allocated : {a}')
        print(f'free      : {f}\n')

        iteration_loss = 0.0
        episode_grads = []  # NI store the gradients of an episode for all languages

        """Inner adaptation loop"""
        torch.cuda.empty_cache()
        for j, task_generator in enumerate(training_tasks):

            language_grads = torch.Tensor()
            learner = meta_m.clone()

            try:  # Sample a batch

                try:
                    support_set = next(task_generator)

                except StopIteration:
                    training_tasks[j] = restart_iter(task_generator, args)
                    task_generator = training_tasks[j]
                    support_set = next(task_generator)  # Sample from new iter

                support_set = move_to_device(support_set, device)
                if SKIP_UPDATE == 0.0 or torch.rand(1) > SKIP_UPDATE:

                    for mini_epoch in range(UPDATES):

                        torch.cuda.empty_cache()

                        # build in a check if the CUDA runs out, and if so changes the support set
                        inner_loss = learner.forward(**support_set)["loss"]

                        # NI - The following two lines implement learning.adapt. See our_maml.py for details
                        # learner.adapt(inner_loss, first_order=True)
                        grads = autograd.grad(inner_loss, learner.parameters(), create_graph=False, retain_graph=False, allow_unused=True)
                        maml_update(learner, lr=args.inner_lr_decoder, lr_small=args.inner_lr_bert, grads=grads)

                        del inner_loss
                        torch.cuda.empty_cache()

                        if (iteration+1) % args.save_every == 0:  # NI
                            new_grads = [g.detach().cpu().reshape(-1) for g in grads if type(g) == torch.Tensor]  # filters out None grads
                            grads_to_save = torch.hstack(new_grads).detach().cpu()  # getting all the parameters
                            language_grads = torch.cat([language_grads.cpu(), grads_to_save], dim=-1)  # Updates * grad_len in the last update

                            del grads_to_save
                            del new_grads
                            torch.cuda.empty_cache()

                del support_set
                torch.cuda.empty_cache()

                if (iteration+1) % args.save_every == 0:  # NI
                    language_grads = language_grads.reshape(-1, UPDATES)  # setup for taking the average

                    if args.accumulation_mode == "mean":
                        language_grads = torch.mean(language_grads, dim=1)  # number of gradients x 1
                    else:
                        language_grads = torch.sum(language_grads, dim=1)  # number of gradients x 1

                    episode_grads.append(language_grads.detach().cpu().numpy())

                try:
                    query_set = next(task_generator)
                except StopIteration:  # Exception called if iter reached its end.
                    # We create a new iterator to use instead
                    training_tasks[j] = restart_iter(task_generator,args)
                    task_generator = training_tasks[j]
                    query_set = next(task_generator)

                query_set = move_to_device(query_set, device)
                eval_loss = learner.forward(**query_set)["loss"]
                iteration_loss += eval_loss

                del eval_loss
                del learner
                del query_set
                torch.cuda.empty_cache()

            except RuntimeError:
                print(f'[ERROR]: Encountered a runtime error at iteration {iteration} for training task {j}.',
                      f' Skipping this training task.')
                continue

        if (iteration + 1) % args.save_every == 0:
            epi_grads = np.array(episode_grads)
            print("[INFO]: Calculating cosine similarity matrix ...")
            cos_matrix = cosine_similarity(epi_grads)
            cos_matrices.append(np.array(cos_matrix))
            print("Cos matrices shape", np.array(cos_matrices).shape)

        # NI end
        # Sum up and normalize over all 7 losses
        iteration_loss /= len(training_tasks)
        optimizer.zero_grad()
        iteration_loss.backward()
        optimizer.step()
        scheduler.step()

        # Bookkeeping
        with torch.no_grad():

            print(iteration, "meta", iteration_loss.item())

            with open(META_WRITER, "a") as f:
                f.write(str(iteration) + " meta " + str(iteration_loss.item()))
                f.write("\n")

        del iteration_loss
        torch.cuda.empty_cache()

        if (iteration + 1) % args.save_every == 0:

            for filename in glob.glob(os.path.join(MODEL_VAL_DIR, "model*")):  # remove the previous temp grads
                os.remove(filename)

            backup_path = os.path.join(MODEL_VAL_DIR, "model" + str(iteration + 1) + ".th")
            torch.save(meta_m.module.state_dict(), backup_path)
            last_iter = iteration + 1
        
        # NI - Save the gradients in case OOM occurs
        if (iteration+1) % args.save_every == 0:  # not to slow down a lot

            file_path_ = f"cos_matrices/temp_allGrads_{args.name}_episode_upd{UPDATES}_pretrain{PRETRAIN_LAN}_suppSize{args.support_set_size}_order{args.language_order}_acc_mode{args.accumulation_mode}"
            # Delete the last temp file
            for filename in glob.glob(f"{file_path_}*"): # remove the previoustemp grads
                os.remove(filename) 

            np.save(f"{file_path_}_cos_mat{iteration}", np.array(cos_matrices))
            torch.cuda.empty_cache()

    # Delete the last temp file
    for filename in glob.glob(f"{file_path_}*"):  # remove the previoustemp grads
        os.remove(filename) 

    cos_matrices = np.array(cos_matrices)
    print(f"[INFO]: Saving the similarity matrix with shape {cos_matrices.shape}")
    np.save(f"cos_matrices/allGrads_{args.name}_episode_upd{UPDATES}_pretrain{PRETRAIN_LAN}_suppSize{args.support_set_size}_order{args.language_order}_acc_mode{args.accumulation_mode}_cos_mat{EPISODES}", cos_matrices)

    print("Done training ... archiving three models!")
    for i in [EPISODES]:

        filename = os.path.join(MODEL_VAL_DIR, "model" + str(i) + ".th")

        if os.path.exists(filename):
            save_place = MODEL_VAL_DIR + "/" + str(i)
            subprocess.run(["mv", filename, MODEL_VAL_DIR + "/best.th"])
            subprocess.run(["mkdir", save_place])
            archive_model(MODEL_VAL_DIR, archive_path=save_place)

    subprocess.run(["rm", MODEL_VAL_DIR + "/best.th"])


if __name__ == "__main__":
    main()
