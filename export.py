import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from model import SpeakerEncoder, WrappedModel, ModelHandling
import argparse
import glob


def export_model(args, check=True):
    net = SpeakerEncoder(**vars(args))

    max_iter_size = args.step_size
    model = ModelHandling(
        net, **dict(vars(args), T_max=max_iter_size))

    model_save_path = os.path.join(
        args.output_folder, f"{args.model['name']}/{args.criterion['name']}/model")

    result_save_path = os.path.join(
        args.output_folder, f"{args.model['name']}/{args.criterion['name']}/result")

    # priority: define weight -> best weight -> last weight
    if args.initial_model_infer:
        chosen_model_state = args.initial_model_infer
    elif os.path.exists(f'{model_save_path}/best_state.pt'):
        chosen_model_state = f'{model_save_path}/best_state.pt'
    else:
        model_files = glob.glob(os.path.join(
            model_save_path, 'model_state_*.model'))
        chosen_model_state = model_files[-1]
    print("Export from ", chosen_model_state)

    model.export_onnx(chosen_model_state, check=check)
