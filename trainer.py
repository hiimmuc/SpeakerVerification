#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import time
import os
import argparse
import socket
import yaml
import numpy
import pdb
import torch
import glob
import zipfile
import warnings
import datetime
import glob
import os
import sys
import time
import subprocess

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from callbacks.earlyStopping import *
from dataloader import train_data_loader
from model import SpeakerEncoder, WrappedModel, ModelHandling
from utils import tuneThresholdfromScore, read_log_file, plot_from_file, cprint
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
###

# Try to import NSML
try:
    import nsml
    from nsml import HAS_DATASET, DATASET_PATH, PARALLEL_WORLD, PARALLEL_PORTS, MY_RANK
    from nsml import NSML_NFS_OUTPUT, SESSION_NAME
except:
    pass

warnings.simplefilter("ignore")


def train(gpu, ngpus_per_node, args):
    args.gpu = gpu
    # paths
    model_save_path = os.path.join(
        args.output_folder, f"{args.model['name']}/{args.criterion['name']}/model")

    result_save_path = os.path.join(
        args.output_folder, f"{args.model['name']}/{args.criterion['name']}/result")

    # TensorBoard
    if args.gpu == 0:
        writer = SummaryWriter(log_dir=f"{result_save_path}/runs")
    os.makedirs(f"{result_save_path}/runs", exist_ok=True)

    # init parameters
    it = 1
    min_loss = float("inf")
    min_eer = float("inf")

    # Initialise data loader
    train_loader = train_data_loader(args)
    max_iter_size = len(train_loader) // args.dataloader_options['nPerSpeaker']

    # Load models
    s = SpeakerEncoder(**vars(args))
    # init parallelism create net-> load weight -> add to parallelism

    # NOTE: Data parallelism for multi-gpu in BETA
    try:
        if torch.cuda.device_count() > 1 and args.data_parallel:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            s = nn.DataParallel(
                s, device_ids=[i for i in range(torch.cuda.device_count())])
        s = s.to(args.device)
    except Exception as e:
        s = WrappedModel(s).to(args.device)

    # NOTE: setup distributed data parallelism training
    if args.distributed:
        try:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = args.port

            dist.init_process_group(
                backend=args.distributed_backend,
                world_size=ngpus_per_node, rank=args.gpu)

            torch.cuda.set_device(args.gpu)
            s.cuda(args.gpu)

            s = torch.nn.parallel.DistributedDataParallel(
                s, device_ids=[args.gpu], find_unused_parameters=True)

            print('Loaded the model on GPU {:d}'.format(args.gpu))
        except:
            dist.destroy_process_group()

    else:
        s = WrappedModel(s).cuda(args.gpu)

    speaker_model = ModelHandling(s, **dict(vars(args), T_max=max_iter_size))

    print(f"Using pretrained: {args.pretrained['use']}")

    # load state from log file
    if os.path.isfile(os.path.join(model_save_path, "model_state_log.txt")):
        start_it, start_lr, _ = read_log_file(
            os.path.join(model_save_path, "model_state_log.txt"))
    else:
        start_it = 1
        start_lr = args.lr

    # Load model weights
    model_files = glob.glob(os.path.join(model_save_path, 'model_state_*.pt'))
    model_files.sort()

    # if exists best model load from it and load state from log model file
    prev_model_state = None
    if start_it > 1:
        if os.path.exists(f'{model_save_path}/best_state.pt'):
            prev_model_state = f'{model_save_path}/best_state.pt'
        elif args.save_model_last:
            if os.path.exists(f'{model_save_path}/last_state.pt'):
                prev_model_state = f'{model_save_path}/last_state.pt'
        else:
            prev_model_state = model_files[-1]

        if args.number_of_epochs > start_it:
            it = int(start_it)
        else:
            it = 1

    # NOTE: Priority: defined pretrained > previous state from logger > scratch
    if args.pretrained['use']:
        speaker_model.loadParameters(args.pretrained['path'])
        print("Model %s loaded!" % args.pretrained['path'])
        it = 1
        args.lr = start_lr
    elif prev_model_state:
        speaker_model.loadParameters(prev_model_state)
        print("Model %s loaded from previous state!" % prev_model_state)
        args.lr = start_lr
        it = start_it
    else:
        print("Train model from scratch!")
        it = 1
        start_lr = args.lr

    # Write args to score_file
    settings_file_path = os.path.join(result_save_path, 'settings.txt')
    settings_file = open(settings_file_path, 'a+')
    score_file_path = os.path.join(result_save_path, 'scores.txt')
    score_file = open(score_file_path, "a+")

    # summary settings
    if args.gpu == 0:
        settings_file.write(
            f'\n[TRAIN]------------------{time.strftime("%Y-%m-%d %H:%M:%S")}------------------\n')
        score_file.write(
            f'\n[TRAIN]------------------{time.strftime("%Y-%m-%d %H:%M:%S")}------------------\n')
        # write the settings to settings file
        for items in vars(args):
            settings_file.write('%s %s\n' % (items, vars(args)[items]))
        settings_file.flush()

    # define early stop
    if args.early_stopping:
        early_stopping = EarlyStopping(patience=args.es_patience)

    top_count = 1

    # Training loop

    timer = time.time()
    while True:
        clr = [x['lr'] for x in speaker_model.__optimizer__.param_groups]

        print(time.strftime("%Y-%m-%d %H:%M:%S"), it,
              "[INFO] Training %s with LR %f ---" % (args.model['name'], max(clr)))

        # Train network
        loss, train_acc = speaker_model.fit(loader=train_loader, epoch=it)

        # save best model
        if loss == min(min_loss, loss):
            cprint(
                text=f"[INFO] Loss reduce from {min_loss} to {loss}. Save the best state", fg='y')
            speaker_model.saveParameters(model_save_path + "/best_state.pt")

            speaker_model.saveParameters(model_save_path +
                                         f"/best_state_top{top_count}.pt")
            # to save top 3 of best_state
            top_count = (top_count + 1) if top_count <= 3 else 1
            if args.early_stopping:
                early_stopping.counter = 0  # reset counter of early stopping

        min_loss = min(min_loss, loss)

        # Validate and save
        if args.test_interval > 0 and it % args.test_interval == 0:
            sc, lab, _ = speaker_model.evaluateFromList(args.test_list,
                                                        cohorts_path=None,
                                                        eval_frames=args.valid_annotation)
            result = tuneThresholdfromScore(sc, lab, [1, 0.1])['roc']

            min_eer = min(min_eer, result[1])

            print("[INFO] Evaluating ",
                  time.strftime("%H:%M:%S"),
                  "LR %f, TEER/TAcc %2.2f, TLOSS %f, VEER %2.4f, MINEER %2.4f" %
                  (max(clr), train_acc, loss, result[1], min_eer))
            score_file.write(
                "IT %d, LR %f, TEER/TAcc %2.2f, TLOSS %f, VEER %2.4f, MINEER %2.4f\n"
                % (it, max(clr), train_acc, loss, result[1], min_eer))

            with open(os.path.join(model_save_path, "model_state_log.txt"), 'w+') as log_file:
                log_file.write(f"Epoch:{it}, LR:{max(clr)}, EER: {0}")

            score_file.flush()

            plot_from_file(result_save_path, show=False)

        else:
            # test interval < 0 -> train non stop
            # print("[INFO] Training at", time.strftime("%H:%M:%S"), "LR %f, Accuracy: %2.2f, Loss: %f \n" % (max(clr), train_acc, loss))
            score_file.write("IT %d, LR %f, TEER/TAcc %2.2f, TLOSS %f\n" %
                             (it, max(clr), train_acc, loss))

            with open(os.path.join(model_save_path, "model_state_log.txt"), 'w+') as log_file:
                log_file.write(f"Epoch:{it}, LR:{max(clr)}, EER: {0}")

            score_file.flush()

            plot_from_file(result_save_path, show=False)

        # NOTE: consider save last state only or not, save only eer as the checkpoint for iterations
        if args.save_model_last:
            s.saveParameters(model_save_path + "/last_state.pt")
        else:
            s.saveParameters(model_save_path + "/model_state_%06d.pt" % it)

        if ("nsml" in sys.modules) and args.gpu == 0:
            training_report = {}
            training_report["summary"] = True
            training_report["epoch"] = it
            training_report["step"] = it
            training_report["train_loss"] = loss

            nsml.report(**training_report)

        if it >= args.number_of_epochs:
            if args.gpu == 0:
                score_file.close()
                settings_file.close()
                writer.close()
            sys.exit(1)

        if args.early_stopping:
            early_stopping(loss)
            if early_stopping.early_stop:
                if args.gpu == 0:
                    score_file.close()
                    settings_file.close()
                    writer.close()
                sys.exit(1)

        if ((time.time() - timer) // 60) % args.ckpt_interval_minutes == 0:
            # save every N mins and keep only top 3
            current_time = 'Day_hour_min'
            ckpt_list = glob.glob(model_save_path, '/ckpt_*')
            if len(ckpt_list) == 3:
                ckpt_list.sort()
                subprocess.call(f'rm -f {ckpt_list[-1]}', shell=True)
            speaker_model.saveParameters(
                model_save_path + f"/ckpt_{current_time}.pt")

        if args.gpu == 0:
            writer.add_scalar('Loss/train', loss, it)
            writer.add_scalar('Accuracy/train', train_acc, it)
            writer.add_scalar('Params/learning_rate', max(clr), it)
        it += 1


# ============================ END =============================
