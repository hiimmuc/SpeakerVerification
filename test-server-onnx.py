import base64
import enum
import io
import json
import os
import time
from argparse import Namespace
from json import dumps
from pathlib import Path

import numpy as np
import torch
from flask import (Flask, Markup, flash, jsonify, redirect, render_template,
                   request, send_file, send_from_directory, url_for)
from flask_restful import Api, Resource
from pydub import AudioSegment
from werkzeug.utils import secure_filename
import soundfile as sf
from processing.audio_loader import *

from model import SpeakerEncoder, WrappedModel, ModelHandling
import onnxruntime as onnxrt
from utils import (read_config, cprint)
from server_utils import *

# check log folder exists
log_service_root = str(Path('log_service/'))
os.makedirs(log_service_root, exist_ok=True)

# ==================================================load Model========================================
norm_mode = 'uniform'
base_threshold = 0.5
compare_threshold = 0.6

threshold = 0.30186375975608826
model_path_onnx = str(Path('backup/20220306/save/Raw_ECAPA/ARmSoftmax/model/model_eval_Raw_ECAPA.onnx'))
model_path = str(Path('backup/1001/Raw_ECAPA/ARmSoftmax/model/best_state_top4.pt'))

config_path = str(Path('yaml/verification.yaml'))
print("\n<<>> Loaded from:", model_path, "with threshold:", threshold)

# read config and load model
args = read_config(config_path)
args = Namespace(**args)

sr = args.audio_spec['sample_rate']
num_eval = args.num_eval
normalize=True
##
t0 = time.time()
net = WrappedModel(SpeakerEncoder(**vars(args)))
max_iter_size = args.step_size
speaker_model = ModelHandling(
        net, **dict(vars(args), T_max=max_iter_size))
speaker_model.loadParameters(model_path, show_error=False)
speaker_model.__model__.eval()


if __name__ == '__main__':
    def to_numpy(tensor):
        if not torch.is_tensor(tensor):
            tensor = torch.FloatTensor(tensor)
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    print("load audio for inference")
    sample_path = 'dataset/train_data/samples/1_Khanh_An_8000.wav'
    inp = torch.FloatTensor(loadWAV(sample_path,
                                    args.audio_spec,
                                    evalmode=True,
                                    augment=False,
                                    augment_options=[],
                                    num_eval=args.num_eval,
                                    random_chunk=False))
    print("starting inference")
    t = time.time()
    onnx_session = onnxrt.InferenceSession(model_path_onnx, providers=['CUDAExecutionProvider'])
    onnx_inputs = {onnx_session.get_inputs()[0].name: to_numpy(inp)}
    onnx_output = onnx_session.run(None, onnx_inputs)
    print("Done inference", time.time() - t)
    print(onnx_output[0].shape)
    
    t = time.time()    
    speaker_model.__model__.eval()
    com_emb = speaker_model.embed_utterance(sample_path, num_eval=args.num_eval, normalize=False).detach().cpu().numpy()
    print("Done inference", time.time() - t)
    print(com_emb.shape)
    
    np.testing.assert_allclose(to_numpy(com_emb), onnx_output[0], rtol=1e-03, atol=1e-05)
    
    