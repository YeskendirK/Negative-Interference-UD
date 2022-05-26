# -*- coding: utf-8 -*-
"""
This file Meta-Trains on 7 languages
And validates on Bulgarian
"""
from get_language_dataset import get_language_dataset

from get_default_params import get_params
from udify import util
from ourmaml import MAML, maml_update
from udify.predictors import predictor
from allennlp.common.util import prepare_environment
from allennlp.models.model import Model
from allennlp.models.archival import archive_model
import allennlp
from schedulers import get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import LambdaLR
from torch import autograd
from torch.optim import Adam
import torch
import numpy as np
import argparse
import subprocess
import json
import sys
import os, glob
import random
from pathlib import Path
import naming_conventions
from allennlp.nn import util


from allennlp.nn.util import move_to_device

from sklearn.metrics.pairwise import cosine_similarity
#Some commennt

sys.stdout.reconfigure(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--skip_update", default=0, type=float, help="Skip update on the support set"
    )
    parser.add_argument("--seed", default=9999, type=int, help="Set seed")
    # Throwing an argparse error
    # parser.add_argument(
    #     "--skip_update", default=0, type=float, help="Skip update on the support set"
    # )
    parser.add_argument("--resume", default=None, type=str, help="If you want to resume train_meta from a previous "
                                                                 "checkpoint. A weights file eg "
                                                                 "saved_models/model250.th should be loaded")
    parser.add_argument("--checkpointep", default=0, type=int, help="By default the iteration starts from 0, "
                                                                    "if specified(Usually when --resume is used), iteration count begins from this number")
    parser.add_argument(
        "--support_set_size", default=32, type=int, help="Support set size"
    )
    parser.add_argument(
        "--maml",
        default=False,
        type=bool,
        help="Do MAML instead of XMAML, that is, include English as an auxiliary task if flag is set and start from scratch",
    )
    parser.add_argument("--addenglish", default=False,
                        type=bool, help="Add English as a task")
    parser.add_argument("--notaddhindi", default=False,
                        type=bool, help="Skip Hindi as a task")

    parser.add_argument("--notadditalian", default=False,
                        type=bool, help="Skip Italian as a task")
    parser.add_argument("--notaddczech", default=False,
                        type=bool, help="Skip Czech as a task")

    parser.add_argument("--episodes", default=900,
                        type=int, help="Amount of episodes")
    parser.add_argument(
        "--updates", default=5, type=int, help="Amount of inner loop updates"
    )
    parser.add_argument("--name", default="", type=str, help="Name to add")
    parser.add_argument(
        "--meta_lr_decoder",
        default=None,
        type=float,
        help="Meta adaptation LR for the decoder",
    )
    parser.add_argument(
        "--meta_lr_bert", default=None, type=float, help="Meta adaptation LR for BERT"
    )
    parser.add_argument(
        "--inner_lr_decoder",
        default=None,
        type=float,
        help="Inner learner LR for the decoder",
    )
    parser.add_argument(
        "--inner_lr_bert", default=None, type=float, help="Inner learner LR for BERT"
    )

    parser.add_argument(
        "--model_dir",
        default=None,
        type=str,
        help="Directory from where to start training. Should be a 'clean' model for MAML and a pretrained model for X-MAML.",
    )

    parser.add_argument("--save_grads_every", default=10, type=int,
                        help="Save the gradient conflicts every save_every episodes ")
    parser.add_argument("--accumulation_mode", default="sum", type=str,
                        help="What gradient accumulation strategy to use", choices=["mean", "sum"])
    args = parser.parse_args()

    device_num = 0 if torch.cuda.is_available() else -1 #torch.device('cuda')

    training_tasks = []
    train_languages = np.array(naming_conventions.train_languages)
    train_languages_lowercase = np.array(naming_conventions.train_languages_lowercase)

    hindi_indices = [0, 1, 2, 3, 4, 6]
    italian_indices = [0, 1, 3, 4, 5, 6]
    czech_indices = [0, 2, 3, 4, 5, 6]
    if args.notaddhindi:
        train_languages = train_languages[hindi_indices]
        train_languages_lowercase = train_languages_lowercase[hindi_indices]
    elif args.notaddczech:
        train_languages = train_languages[czech_indices]
        train_languages_lowercase = train_languages_lowercase[czech_indices]
    elif args.notadditalian:
        train_languages = train_languages[italian_indices]
        train_languages_lowercase = train_languages_lowercase[italian_indices]

    for lan, lan_l in zip(train_languages, train_languages_lowercase):
        training_tasks.append(get_language_dataset(
            lan, lan_l, seed=args.seed, support_set_size=args.support_set_size))

    # Setting parameters
    DOING_MAML = args.maml
    if DOING_MAML or args.addenglish:
        # Get another training task
        training_tasks.append(
            get_language_dataset(
                "UD_English-EWT",
                "en_ewt-ud",
                seed=args.seed,
                support_set_size=args.support_set_size,
            )
        )
    UPDATES = args.updates
    EPISODES = args.episodes
    INNER_LR_DECODER = args.inner_lr_decoder
    INNER_LR_BERT = args.inner_lr_bert
    META_LR_DECODER = args.meta_lr_decoder
    META_LR_BERT = args.meta_lr_bert
    SKIP_UPDATE = args.skip_update
    PRETRAIN_LAN = args.model_dir.split("/")[-2] if args.model_dir is not None else "NaN"  # says what language we are using

    # Filenames
    MODEL_FILE = (
        args.model_dir
        if args.model_dir is not None
        else (
            "../backup/pretrained/english_expmix_deps_seed2/2020.07.30_18.50.07"
            if not DOING_MAML
            else "logs/english_expmix_tiny_deps2/2020.05.29_17.59.31"
        )
    )

    maml_string = "saved_models/MAML" if DOING_MAML else "saved_models/XMAML"
    param_list = [
        str(z)
        for z in [
            maml_string,
            PRETRAIN_LAN,
            INNER_LR_DECODER,
            INNER_LR_BERT,
            META_LR_DECODER,
            META_LR_BERT,
            UPDATES,
            args.seed,
        ]
    ]
    MODEL_SAVE_NAME = "_".join(param_list)
    MODEL_VAL_DIR = MODEL_SAVE_NAME + args.name
    META_WRITER = MODEL_VAL_DIR + "/meta_results.txt"

    if not os.path.exists(MODEL_VAL_DIR):
        subprocess.run(["mkdir", "-p", MODEL_VAL_DIR]) # Missing the parent tag, saved_models folder doesn't exist by itself
        subprocess.run(["mkdir", MODEL_VAL_DIR + "/performance"])
        subprocess.run(["mkdir", MODEL_VAL_DIR + "/predictions"])
        subprocess.run(["cp", "-r", MODEL_FILE + "/vocabulary", MODEL_VAL_DIR])
        subprocess.run(["cp", MODEL_FILE + "/config.json", MODEL_VAL_DIR])

    with open(META_WRITER, "w") as f:
        f.write("Model ready\n")

    # Loading the model
    train_params = get_params("metalearning", args.seed)
    prepare_environment(train_params)
    m = Model.load(
        train_params,
        MODEL_FILE,
    )
    meta_m = MAML(
        m, INNER_LR_DECODER, INNER_LR_BERT, first_order=True, allow_unused=True
    )
    if args.resume:
        model_state = torch.load(args.resume)
        meta_m.module.load_state_dict(model_state)
    meta_m.cuda()
    optimizer = Adam(
        [
            {
                "params": meta_m.module.text_field_embedder.parameters(),
                "lr": META_LR_BERT,
            },
            {"params": meta_m.module.decoders.parameters(), "lr": META_LR_DECODER},
            {"params": meta_m.module.scalar_mix.parameters(), "lr": META_LR_DECODER},
        ],
        META_LR_DECODER,
    )  # , weight_decay=0.01)

    scheduler = get_cosine_schedule_with_warmup(optimizer, 0.1*EPISODES, EPISODES)

    def restart_iter(language, language_lowercase_, args_):
        return get_language_dataset(language, language_lowercase_,
                                    seed=args_.seed, support_set_size=args_.support_set_size)

    print(f"BEGINNING THE META TRAIN FROM EPISODE = {args.checkpointep}")
    # compute graident conflicts
    cos_matrices = []
    lowest_iteration_loss = 1e10
    for iteration in range(args.checkpointep, EPISODES):

        # print(f"[INFO]: Starting episode {iteration}\n", flush=True)


        iteration_loss = 0.0
        episode_grads = []  # store the gradients of an episode for all languages

        """Zip and enumerate everything we need. Zip by itself doesn't slow down anything, so a quick fix for
            the dataset reload issue.
        """
        num_grad_samples = min(4, UPDATES)
        epochs_grad_conflict = random.sample(range(UPDATES), k=num_grad_samples)  # random.randint(0, UPDATES-1)
        for j, (task_generator, train_lang, train_lang_low) in \
                enumerate(zip(training_tasks, train_languages, train_languages_lowercase)):
            learner = meta_m.clone(first_order=True)
            language_grads = torch.Tensor()

            try:
                support_set = next(task_generator)[0]

            except StopIteration as e:
                print(f"Exception raised - {e} in support set. Task generator is empty")
                training_tasks[j] = restart_iter(train_lang, train_lang_low, args)
                task_generator = training_tasks[j]
                support_set = next(task_generator)[0]
            print("SUPPORT SET metadata len = ", len(support_set['metadata']))
            # print(support_set['metadata'])
            print("-"*20)
            print("-" * 20)
            # support_set = move_to_device(support_set, device_num)
            if SKIP_UPDATE == 0.0 or torch.rand(1) > SKIP_UPDATE:

                for mini_epoch in range(UPDATES):
                    inner_loss = learner.forward(**support_set)["loss"]

                    # compute graident conflicts
                    grads = autograd.grad(inner_loss, learner.parameters(), create_graph=False, retain_graph=False,
                                          allow_unused=True)
                    maml_update(learner, lr=args.inner_lr_decoder, lr_small=args.inner_lr_bert, grads=grads)
                    #learner.adapt(inner_loss, first_order=True)
                    del inner_loss
                    torch.cuda.empty_cache()

                    # compute graident conflicts
                    compute_grad_conflict = mini_epoch in epochs_grad_conflict

                    if (iteration + 1) % args.save_grads_every == 0 and compute_grad_conflict:  # NI
                        new_grads = [g.detach().cpu().reshape(-1) for g in grads if
                                     type(g) == torch.Tensor]  # filters out None grads
                        grads_to_save = torch.hstack(new_grads).detach().cpu()  # getting all the parameters
                        # grads_to_save = torch.cat(new_grads, dim=-1).detach().cpu()  # getting all the parameters
                        language_grads = torch.cat([language_grads.cpu(), grads_to_save],
                                                   dim=-1)  # Updates * grad_len in the last update


                        del grads_to_save
                        del new_grads
                        torch.cuda.empty_cache()


            del support_set
            torch.cuda.empty_cache()

            if (iteration + 1) % args.save_grads_every == 0:  # NI
                language_grads = language_grads.reshape(-1, num_grad_samples)  # setup for taking the average

                if args.accumulation_mode == "mean":
                    language_grads = torch.mean(language_grads, dim=1)  # number of gradients x 1
                else:
                    language_grads = torch.sum(language_grads, dim=1)  # number of gradients x 1

                episode_grads.append(language_grads.detach().cpu().numpy())

            try:
                query_set = next(task_generator)[0]
            except StopIteration as e:
                print(f"Exception raised -  {e} in query_set")
                training_tasks[j] = restart_iter(train_lang, train_lang_low, args)
                task_generator = training_tasks[j]
                query_set = next(task_generator)[0]
            print("QUERY SET len = ", len(query_set['metadata']))
            # print(query_set)
            print("="*20)
            # query_set = move_to_device(query_set, device_num)
            eval_loss = learner.forward(**query_set)["loss"]
            iteration_loss += eval_loss
            del eval_loss
            del learner
            del query_set
            torch.cuda.empty_cache()

        if (iteration + 1) % args.save_grads_every == 0:
            epi_grads = np.array(episode_grads)
            print(epi_grads.shape)
            episode_grads_shapes = [x.shape for x in episode_grads]
            print(episode_grads_shapes)
            print(len(episode_grads))
            print("[INFO]: Calculating cosine similarity matrix ...")
            cos_matrix = cosine_similarity(epi_grads)
            cos_matrices.append(np.array(cos_matrix))

            print("Cos matrices shape", np.array(cos_matrices).shape)


        # Sum up and normalize over all 7 losses
        iteration_loss /= len(training_tasks)
        optimizer.zero_grad()
        iteration_loss.backward()
        optimizer.step()
        scheduler.step()

        # Save best model
        if iteration_loss < lowest_iteration_loss:
            best_model_path = os.path.join(MODEL_VAL_DIR, "best_model.th")
            print(f"BEST Model updated on {iteration} epoch")
            print(f"Min. Loss decreased from {lowest_iteration_loss} to {iteration_loss}")
            torch.save(meta_m.module.state_dict(), best_model_path)

            lowest_iteration_loss = iteration_loss

        # Bookkeeping
        with torch.no_grad():
            print(iteration, "meta", iteration_loss.item())
            with open(META_WRITER, "a") as f:
                f.write(str(iteration) + " meta " + str(iteration_loss.item()))
                f.write("\n")
        del iteration_loss
        torch.cuda.empty_cache()

        # Saving the 250th episode as oom errors are appearing in lisa
        if iteration + 1 in [10, 100, 250, 500, 1500, 2000] and not (
            iteration + 1 == 500 and DOING_MAML
        ):
            backup_path = os.path.join(
                MODEL_VAL_DIR, "model" + str(iteration + 1) + ".th"
            )
            torch.save(meta_m.module.state_dict(), backup_path)

        # NI - Save the gradients in case OOM occurs
        if (iteration + 1) % args.save_grads_every == 0:  # not to slow down a lot
            cos_dir = f"cos_matrices/{args.name}"
            Path(cos_dir).mkdir(parents=True, exist_ok=True)
            file_path_ = f"cos_matrices/{args.name}/temp_allGrads_{args.name}_episode_upd{UPDATES}_pretrain{PRETRAIN_LAN}_" \
                         f"suppSize{args.support_set_size}_acc_mode{args.accumulation_mode}_iter{iteration + 1}"
            # Delete the last temp file
            for filename in glob.glob(f"{file_path_}*"):  # remove the previoustemp grads
                os.remove(filename)

            np.save(f"{file_path_}_cos_mat{iteration}", np.array(cos_matrices))
            torch.cuda.empty_cache()

    cos_matrices = np.array(cos_matrices)
    print(f"[INFO]: Saving the similarity matrix with shape {cos_matrices.shape}")
    cos_dir = f"cos_matrices/{args.name}"
    Path(cos_dir).mkdir(parents=True, exist_ok=True)
    np.save(
        f"cos_matrices/{args.name}/allGrads_{args.name}_episode_upd{UPDATES}_pretrain{PRETRAIN_LAN}_suppSize{args.support_set_size}_acc_mode{args.accumulation_mode}_cos_mat{EPISODES}",
        cos_matrices)


    print("Done training ... archiving three models!")
    for i in [10, 100, 250, 500, 600, 900, 1200, 1500, 1800, 2000, 1500]:
        filename = os.path.join(MODEL_VAL_DIR, "model" + str(i) + ".th")
        if os.path.exists(filename):
            save_place = MODEL_VAL_DIR + "/" + str(i)
            subprocess.run(["mv", filename, MODEL_VAL_DIR + "/best.th"])
            subprocess.run(["mkdir", save_place])
            archive_model(
                MODEL_VAL_DIR,
                # files_to_archive=train_params.files_to_archive,
                archive_path=save_place,
            )
            print("archieved to save_place: ", save_place)
    best_model_filename = os.path.join(MODEL_VAL_DIR, "best_model.th")

    # Archive best model
    if os.path.exists(best_model_filename):
        save_place = MODEL_VAL_DIR + "/" + 'best'
        subprocess.run(["mv", best_model_filename, MODEL_VAL_DIR + "/best.th"])
        subprocess.run(["mkdir", save_place])
        archive_model(
            MODEL_VAL_DIR,
            files_to_archive=train_params.files_to_archive,
            archive_path=save_place,
        )
        print("BEST model archieved to save_place: ", save_place)

    subprocess.run(["rm", MODEL_VAL_DIR + "/best.th"])


if __name__ == "__main__":
    main()
