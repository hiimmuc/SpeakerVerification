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
from model import SpeakerNet, WrappedModel
from utils import tuneThresholdfromScore, read_log_file, plot_from_file, cprint

from torch.utils.tensorboard import SummaryWriter
###


def train(gpu, ngpus_per_node, args):
    model_save_path = os.path.join(
        args.output_folder, f"{args.model['name']}/{args.criterion['name']}/model")

    result_save_path = os.path.join(
        args.output_folder, f"{args.model['name']}/{args.criterion['name']}/result")

    # TensorBoard
    writer = SummaryWriter(log_dir=f"{result_save_path}/runs")
    os.makedirs(f"{result_save_path}/runs", exist_ok=True)

    # init parameters
    it = 1
    min_loss = float("inf")
    min_eer = float("inf")

    # Initialise data loader
    train_loader = train_data_loader(args.train_annotation, **vars(args))
    max_iter_size = len(train_loader) // args.nPerSpeaker

    # Load models
    s = SpeakerNet(**dict(vars(args), T_max=max_iter_size))
    # setup multi gpus
    # init parallelism create net-> load weight -> add to parallelism
    try:
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            s.__S__ = nn.DataParallel(
                s.__S__, device_ids=[i for i in range(torch.cuda.device_count())])
        s.__S__ = s.__S__.to(args.device)
    except Exception as e:
        print(e)
        s.__S__ = s.__S__.to(args.device)

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
        s.loadParameters(args.pretrained['path'])
        print("Model %s loaded!" % args.pretrained['path'])
        it = 1
        args.lr = start_lr
    elif prev_model_state:
        s.loadParameters(prev_model_state)
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
        clr = [x['lr'] for x in s.__optimizer__.param_groups]

        print(time.strftime("%Y-%m-%d %H:%M:%S"), it,
              "[INFO] Training %s with LR %f ---" % (args.model['name'], max(clr)))

        # Train network
        loss, train_acc = s.fit(loader=train_loader, epoch=it)

        # save best model
        if loss == min(min_loss, loss):
            cprint(
                text=f"[INFO] Loss reduce from {min_loss} to {loss}. Save the best state", fg='y')
            s.saveParameters(model_save_path + "/best_state.pt")

            s.saveParameters(model_save_path +
                             f"/best_state_top{top_count}.pt")
            # to save top 3 of best_state
            top_count = (top_count + 1) if top_count <= 3 else 1
            if args.early_stopping:
                early_stopping.counter = 0  # reset counter of early stopping

        min_loss = min(min_loss, loss)

        # Validate and save
        if args.test_interval > 0 and it % args.test_interval == 0:
            sc, lab, _ = s.evaluateFromList(args.test_list,
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

        if it >= args.number_of_epochs:
            score_file.close()
            settings_file.close()
            writer.close()
            sys.exit(1)

        if args.early_stopping:
            early_stopping(loss)
            if early_stopping.early_stop:
                score_file.close()
                settings_file.close()
                writer.close()
                sys.exit(1)

        if (time.time() - timer) // 60 >= args.ckpt_interval_minutes:
            # save every N mins and keep only top 3
            current_time = 'Day_hour_min'
            ckpt_list = glob.glob(model_save_path, '/ckpt_*')
            if len(ckpt_list) == 3:
                ckpt_list.sort()
                subprocess.call(f'rm -f {ckpt_list[-1]}', shell=True)
            s.saveParameters(model_save_path + f"/ckpt_{current_time}.pt")

        writer.add_scalar('Loss/train', loss, it)
        writer.add_scalar('Accuracy/train', train_acc, it)
        writer.add_scalar('Params/learning_rate', max(clr), it)
        it += 1


# ============================ END =============================
