# benchmark dataset:
import itertools
import os
import csv
import time
import glob

import argparse
from pathlib import Path
import numpy as np
from pydub import AudioSegment
from model import SpeakerEncoder, WrappedModel, ModelHandling

from tqdm.auto import tqdm
from utils import similarity_measure
from utils import read_config, tuneThresholdfromScore



def all_pairs(lst):
    return list(itertools.combinations(lst, 2))


def check_matching(ref_emb, com_emb, threshold=0.5):
    score = similarity_measure(method='cosine',ref= ref_emb, com=com_emb)
    ratio = threshold / 0.5
    result = (score / ratio) if (score / ratio) < 1 else 1
    matching = result > 0.5
    return matching, result


def benchmark_dataset_by_model(model_path=None,
                               config_path=None,
                               threshold=0.38405078649520874,
                               root='dataset/train-data/',
                               save_file='benchmark_result.txt'):
    # load model
    args = read_config(config_path)
    args = argparse.Namespace(**args)
    

    t0 = time.time()
    net = WrappedModel(SpeakerEncoder(**vars(args)))
    max_iter_size = args.step_size
    model = ModelHandling(net, **dict(vars(args), T_max=max_iter_size))
    model.loadParameters(model_path, show_error=False)
    model.__model__.eval()
    print("Model Loaded time: ", time.time() - t0)

    # ===================================================
    folders = glob.glob(str(Path(root, '*')))

    for folder in tqdm(folders[:]):
        filepaths = glob.glob(f"{folder}/*.wav")

        pairs = all_pairs(filepaths)
        files_emb_dict = {}
        imposters = {}

        for fn in filepaths:
            emb = model.embed_utterance(fn,
                                        num_eval=20,
                                        normalize=True)
            if fn not in files_emb_dict:
                files_emb_dict[fn] = emb

        for pair in pairs:
            match, score = check_matching(
                files_emb_dict[pair[0]], files_emb_dict[pair[1]], threshold)
            if not match:
                if pair[0] not in imposters:
                    imposters[pair[0]] = 0
                if pair[1] not in imposters:
                    imposters[pair[1]] = 0
                imposters[pair[0]] += 1
                imposters[pair[1]] += 1
        imposters_list = [k for k, v in imposters.items() if v > 0]

        with open(save_file, 'a+') as f:
            if len(imposters_list) > 0:
                f.write(f"Folder:{folder}\n")
                for imp in sorted(imposters_list):
                    f.write(f"[{imposters[imp]}/{len(filepaths)}] - {imp}\n")
                f.write("//================//\n")

                
if __name__ == '__main__':
    model_path = str(Path('backup/2811/save/Raw_ECAPA_hype/AAmSoftmaxAP/model/best_state.pt'))
    config_path=str(Path('backup/2811/save/Raw_ECAPA_hype/AAmSoftmaxAP/config/configuration.yaml'))
    threshold=0.4
    root='dataset/train_data/gannhan_bm/wav/'
    save_file='benchmark_result.txt'
    
    benchmark_dataset_by_model(model_path, config_path, threshold, root, save_file)