import os
import csv
import importlib
import random
import sys
import time
from pathlib import Path
import time
import os
import itertools
import shutil
import importlib
import numpy as np
import onnx
import onnxruntime as onnxrt
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm.auto import tqdm
from processing.audio_loader import loadWAV
from utils import (similarity_measure, cprint)
from torch.cuda.amp import autocast, GradScaler

class WrappedModel(nn.Module):

    ## The purpose of this wrapper is to make the model structure consistent between single and multi-GPU

    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.module = model

    def forward(self, x, label=None):
        return self.module(x, label)


class SpeakerEncoder(nn.Module):
    
    def __init__(self, model, criterion, classifier, optimizer, features , device, include_top=False, ** kwargs) -> None:
        super(SpeakerEncoder, self).__init__()
        self.model = model
        self.criterion = criterion
        self.classifier = classifier
        self.optimizer = optimizer
        self.device = device
        self.n_mels = kwargs['n_mels']
        self.augment = kwargs['augment']
        self.augment_chain = kwargs['augment_options']['augment_chain']
        
        if features.lower() in ['mfcc', 'melspectrogram']:
            Features_extractor = importlib.import_module(
                'models.FeatureExtraction.feature').__getattribute__(f"{features.lower()}")
            self.compute_features = Features_extractor(**kwargs).to(self.device)
        else:
            Features_extractor = None 
            self.compute_features = None   
        
        
        SpeakerNetModel = importlib.import_module(
            'models.' + self.model['name']).__getattribute__('MainModel')
        self.__S__ = SpeakerNetModel(nOut=self.model['nOut'],
                                     **kwargs).to(self.device)
        # # necessaries:
        # self.aug = kwargs['augment']
        # self.aug_chain = kwargs['augment_chain']
        # nOut n_mels
        
        
        LossFunction = importlib.import_module(
            'losses.' + self.criterion['name']).__getattribute__(f"{self.criterion['name']}")

        self.__L__ = LossFunction(nOut=self.model['nOut'], 
                                  margin=self.criterion['margin'], 
                                  scale=self.criterion['scale'],
                                  **kwargs).to(self.device)
        #NOTE: nClasses is defined
        self.test_normalize = self.__L__.test_normalize
        
        self.nPerSpeaker = kwargs['dataloader_options']['nPerSpeaker']
        
        self.include_top = include_top
        
        if self.include_top:
            self.fc = nn.Linear(
                self.classifier['input_size'], self.classifier['out_neurons'])

        ####
        nb_params = sum([param.view(-1).size()[0]
                        for param in self.__S__.parameters()])
        print(f"Initialize encoder model {self.model['name']}: {nb_params:,} params")
        print("Embedding normalize: ", self.__L__.test_normalize)
        
    def forward(self, data, label=None):
        # data = data.cuda()
        
        feat = []
        # forward n utterances per speaker and stack the output
        for inp in data:
            if self.compute_features is not None:
                inp = self.compute_features(inp)
            outp = self.__S__(inp.to(self.device))
            feat.append(outp)

        feat = torch.stack(feat, dim=1).squeeze()
                
        ## Beta for the recognition task
        # if self.include_top:
        #     outp = self.fc(self.__S__.forward(data))
        # else:
        #     outp = self.__S__.forward(data)

        if label == None:
            return feat
        else:
            nloss, prec1 = self.__L__.forward(feat, label)
            return nloss, prec1
                


class ModelHandling(object):
    def __init__(self, encoder_model, 
                 optimizer='adam', 
                 callbacks='steplr',  
                 device='cuda',
                 gpu=0,  mixedprec=False, **kwargs):
        
        # take only args
        super(ModelHandling, self).__init__()
        
        self.kwargs = kwargs
        self.device = torch.device(device)
        self.save_path = self.kwargs['save_folder']
        
        self.T_max = 0 if 'T_max' not in self.kwargs else self.kwargs['T_max']

        self.__model__ = encoder_model
        
        self.scaler = GradScaler()

        self.gpu = gpu

        self.mixedprec = mixedprec
        
        Optimizer = importlib.import_module(
            'optimizer.' + optimizer['name']).__getattribute__(f"{optimizer['name']}")
        self.__optimizer__ = Optimizer(self.__model__.parameters(),
                                       weight_decay=optimizer['weight_decay'],
                                       lr_decay=optimizer['lr_decay'],
                                       **kwargs)

        self.callback = callbacks

        if self.callback['name'] in ['steplr', 'cosinelr', 'cycliclr']:
            Scheduler = importlib.import_module(
                'callbacks.torch_callbacks').__getattribute__(f"{self.callback['name'].lower()}")
            self.__scheduler__, self.lr_step = Scheduler(self.__optimizer__,                                                          
                                                         lr_decay=optimizer['lr_decay'], **dict(kwargs, T_max=self.T_max))
        elif self.callback['name'] == 'reduceOnPlateau':
            Scheduler = importlib.import_module(
                'callbacks.' + callbacks).__getattribute__('LRScheduler')
            self.__scheduler__ = Scheduler(self.__optimizer__, 
                                           step_size=self.callback['step_size'], 
                                           lr_decay=optimizer['lr_decay'], 
                                           patience=self.callback['step_size'], 
                                           min_lr=self.callback['base_lr'], factor=0.95)
            self.lr_step = 'epoch'
        
        assert self.lr_step in ['epoch', 'iteration']

    # ## ===== ===== ===== ===== ===== ===== ===== =====
    # ## Train network
    # ## ===== ===== ===== ===== ===== ===== ===== =====
    def fit(self, loader, epoch=0):
        '''Train

        Args:
            loader (Dataloader): dataloader of training data
            epoch (int, optional): [description]. Defaults to 0.

        Returns:
            tuple: loss and precision
        '''
        self.__model__.train()

        stepsize = loader.batch_size

        counter = 0
        index = 0
        loss = 0
        top1 = 0  # EER or accuracy
                
        loader_bar = tqdm(loader, desc=f">EPOCH_{epoch}", unit="it", colour="green")
        
        for (data, data_label) in loader_bar:
            data = data.transpose(0, 1).to(self.device)
            label = torch.LongTensor(data_label).to(self.device)
            
            self.__model__.zero_grad()
            
            if self.mixedprec:
                with autocast():
                    nloss, prec1 = self.__model__.forward(data, label)
                self.scaler.scale(nloss).backward()
                self.scaler.step(self.__optimizer__)
                self.scaler.update()
            else:
                nloss, prec1 = self.__model__.forward(data, label)
                nloss.mean().backward()
                self.__optimizer__.step()

            loss += nloss.detach().cpu()
            top1 += prec1.detach().cpu()
            counter += 1
            index += stepsize

            # update tqdm bar
            loader_bar.set_postfix(LR=f"{round(float(self.__optimizer__.param_groups[0]['lr']), 8)}", 
                                   TLoss=f"{round(float(loss / counter), 5)}", 
                                   TAcc=f"{round(float(top1 / counter), 3)}%")

            if self.lr_step == 'iteration':
                self.__scheduler__.step()

        # select mode for callbacks
        if self.lr_step == 'epoch' and self.callback not in ['reduceOnPlateau', 'auto']:
            self.__scheduler__.step()

        elif self.callback == 'reduceOnPlateau':
            # reduce on plateau
            self.__scheduler__(loss / counter)

        elif self.callback == 'auto':
            if epoch <= 50:
                self.__scheduler__['rop'](loss / counter)
            else:
                if epoch == 51:
                    cprint("\n[INFO] # Epochs > 50, switch to steplr callback\n========>\n", 'r')
                self.__scheduler__['steplr'].step()

        loss_result = loss / (counter)
        precision = top1 / (counter)

        return loss_result, precision

    def evaluateFromList(self,
                         listfilename,
                         distributed,
                         dataloader_options,
                         cohorts_path='checkpoint/dump_cohorts.npy',
                         num_eval=10,
                         scoring_mode='cosine', **kwargs):
        
        if distributed:
            rank = torch.distributed.get_rank()
        else:
            rank = 0

        self.__model__.eval()

        lines = []
        files = []
        feats = {}

        # Cohorts
        cohorts = None
        if cohorts_path is not None and scoring_mode == 'norm':
            cohorts = np.load(cohorts_path)

        # Read all lines
        with open(listfilename) as listfile:
            lines = listfile.readlines()
            
            # for line in lines:
            #     data = line.split()

            #     # Append random label if missing
            #     if len(data) == 2:
            #         data = [random.randint(0, 1)] + data

            #     files.append(data[1])
            #     files.append(data[2])
            #     lines.append(line)
                
        ## Get a list of unique file names
        files = list(itertools.chain(*[x.strip().split()[-2:] for x in lines]))
        setfiles = list(set(files))
        setfiles.sort()

        test_dataset = test_loader(setfiles, num_eval=num_eval, **kwargs)
        
        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(
                test_dataset, shuffle=False)
        else:
            sampler = None
            
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=dataloader_options['num_workers'],
            drop_last=False,
            sampler=sampler
        )
        
        print(">>>>Evaluation")

        # Save all features to dictionary
        loader_bar = tqdm(
            test_loader, desc=">>>>Reading file: ", unit="files", colour="red")
        for idx, data in enumerate(loader_bar):
            (audio, filename) = data[:][0]
            
            inp1 = torch.FloatTensor(audio).to(self.device)                        

            with torch.no_grad():
                ref_feat = self.__model__.forward(inp1).detach().cpu()
                                
            feats[filename] = ref_feat

        all_scores = []
        all_labels = []
        all_trials = []
        
        if distributed:
            ## Gather features from all GPUs
            feats_all = [None for _ in range(0,torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(feats_all, feats)
        
        if rank == 0:
            # run on main worker
            if distributed:
                feats = feats_all[0]
                for feats_batch in feats_all[1:]:
                    feats.update(feats_batch)
            
            # Read files and compute all scores

            for idx, line in enumerate(tqdm(lines, desc=">>>>Computing files", unit="pairs", colour="MAGENTA")):
                data = line.split()

                # Append random label if missing
                if len(data) == 2:
                    data = [random.randint(0, 1)] + data

                ref_feat = feats[data[1]].to(self.device)
                com_feat = feats[data[2]].to(self.device)

                if self.__model__.test_normalize:
                    ref_feat = F.normalize(ref_feat, p=2, dim=1)
                    com_feat = F.normalize(com_feat, p=2, dim=1)

                # NOTE: distance(cohort = None) for training, normalized score for evaluating and testing
                if cohorts_path is None:
                    dist = F.pairwise_distance(
                        ref_feat.unsqueeze(-1),
                        com_feat.unsqueeze(-1).transpose(
                            0, 2)).detach().cpu().numpy()
                    score = -1 * np.mean(dist)
                else:
                    if scoring_mode == 'norm':
                        score = similarity_measure('zt_norm',
                                                ref_feat,
                                                com_feat,                                                
                                                cohorts,
                                                top=200)
                    elif scoring_mode == 'cosine':
                        score = similarity_measure('cosine',ref_feat, com_feat)
                    elif scoring_mode == 'pnorm' :
                        score = similarity_measure('pnorm', ref_feat, com_feat, p = 2)

                all_scores.append(score)
                all_labels.append(int(data[0]))
                all_trials.append(data[1] + " " + data[2])             
            
        return all_scores, all_labels, all_trials

    def testFromList(self,
                     root,
                     test_list='evaluation_test.txt',
                     thre_score=0.5,
                     cohorts_path=None,
                     num_eval=10,
                     scoring_mode='norm',
                     output_file=None):
        self.__model__.eval()

        lines = []
        files = []
        feats = {}

        # Cohorts
        cohorts = None
        if cohorts_path is not None and scoring_mode == 'norm':
            cohorts = np.load(cohorts_path)
        save_root = self.save_path + f"/{self.model_name}/result"

        data_root = Path(root)
        read_file = Path(test_list)
        if output_file is None:
            output_file = test_list.replace('.txt','_result.txt')
        write_file = Path(save_root, output_file) if os.path.split(output_file)[0] == '' else output_file # add parent dir if not provided
        
        # Read all lines from testfile (read_file)
        print(">>>>TESTING...")
        print(f">>> Threshold: {thre_score}")
        with open(read_file, newline='') as rf:
            spamreader = csv.reader(rf, delimiter=',')
            next(spamreader, None)
            for row in spamreader:
                files.append(row[0])
                files.append(row[1])
                lines.append(row)

        setfiles = list(set(files))
        setfiles.sort()

        # Save all features to feat dictionary
        for idx, filename in enumerate(tqdm(setfiles, desc=">>>>Reading file: ", unit="files", colour="red")):
            audio = loadWAV(filename.replace('\n', ''), evalmode=True, **self.kwargs)
            
            inp1 = torch.FloatTensor(audio).to(self.device)

            with torch.no_grad():
                ref_feat = self.__model__.forward(inp1).detach().cpu()
                
            feats[filename] = ref_feat

        # Read files and compute all scores
        with open(write_file, 'w', newline='') as wf:
            spamwriter = csv.writer(wf, delimiter=',')
            spamwriter.writerow(['audio_1', 'audio_2', 'pred_label' , 'score'])
            for idx, data in enumerate(tqdm(lines, desc=">>>>Computing files", unit="pairs", colour="MAGENTA")):
                ref_feat = feats[data[0]].to(self.device)
                com_feat = feats[data[1]].to(self.device)

                if self.__model__.test_normalize:
                    ref_feat = F.normalize(ref_feat, p=2, dim=1)
                    com_feat = F.normalize(com_feat, p=2, dim=1)

                if cohorts_path is None:
                    dist = F.pairwise_distance(
                        ref_feat.unsqueeze(-1),
                        com_feat.unsqueeze(-1).transpose(
                            0, 2)).detach().cpu().numpy()
                    score = -1 * np.mean(dist)
                else:
                    if scoring_mode == 'norm':
                        score = similarity_measure('zt_norm',ref_feat,
                                                    com_feat,
                                                    cohorts,
                                                    top=200)
                    elif scoring_mode == 'cosine':
                        score = similarity_measure('cosine',ref_feat, com_feat)
                    elif scoring_mode == 'pnorm' :
                        score = similarity_measure('pnorm', ref_feat, com_feat, p = 2)

                pred = '1' if score >= thre_score else '0'
                spamwriter.writerow([data[0], data[1], pred, score])

        
   
    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## preparing the cohort or embeddings of multiple utterances
    ## ===== ===== ===== ===== ===== ===== ===== =====
    def prepare(self,
                save_path=None,
                prepare_type='cohorts',
                num_eval=10,
                eval_frames=100,
                source=None, **kwargs):                    
 
        """
        Prepared 1 of the 2:
        1. Mean L2-normalized embeddings for known speakers.
        2. Cohorts for score normalization.
        """
        
        self.__model__.eval()
        if not source:
            raise "Please provide appropriate source!"
        ########### cohort preparation
        if prepare_type == 'cohorts':
            n_emb_per_spk = 3

            assert isinstance(source, str), "Please provide path to train metadata files"
            read_file = Path(source)

            lines = []
            cohort_spk_files = dict()
            cohort_embedding = dict()

            with open(read_file, 'r') as listfile:
                lines = [line.replace('\n', '') for line in listfile.readlines()]
                
            for line in tqdm(lines, desc='Gathering files...', unit=' files'):
                data = line.split()
                spkID, path = data[:2]
                cohort_spk_files.setdefault(spkID, []).append(path)
                
            for spkID, paths in tqdm(cohort_spk_files.items(), unit=' speakers', desc='Getting speaker embedding'):
                for path in paths[:n_emb_per_spk]:
                    emb = self.embed_utterance(path, eval_frames=eval_frames, num_eval=num_eval, normalize=True)
                    cohort_embedding.setdefault(spkID, []).append(emb)

            cohort_speakers = list(cohort_embedding.keys())
            cohort = np.vstack([np.mean(np.vstack(cohort_embedding[speaker]), axis=0, keepdims=True) 
                                for speaker in cohort_speakers])
            if save_path:
                np.save(save_path, np.array(cohort))
            return True
                
        ############# Embedding preparation
        elif prepare_type == 'embed':
            # Prepare mean L2-normalized embeddings for known speakers.
            # load audio from_path (root path)
            # option 1: from root
            if isinstance(source, str):
                speaker_dirs = [x for x in Path(source).iterdir() if x.is_dir()]
                embeds = None
                classes = {}
                # Save mean features
                for idx, speaker_dir in enumerate(speaker_dirs):
                    classes[idx] = speaker_dir.stem
                    files = list(speaker_dir.glob('*.wav'))
                    mean_embed = None
                    embed = None
                    for f in files:
                        embed = self.embed_utterance(
                            f,
                            eval_frames=eval_frames,
                            num_eval=num_eval,
                            normalize=self.__model__.test_normalize)
                        if mean_embed is None:
                            mean_embed = embed.unsqueeze(0)
                        else:
                            mean_embed = torch.cat(
                                (mean_embed, embed.unsqueeze(0)), 0)
                    mean_embed = torch.mean(mean_embed, dim=0)
                    if embeds is None:
                        embeds = mean_embed.unsqueeze(-1)
                    else:
                        embeds = torch.cat((embeds, mean_embed.unsqueeze(-1)), -1)

                print(embeds.shape)
                # embeds = rearrange(embeds, 'n_class n_sam feat -> n_sam feat n_class')
                if save_path:
                    torch.save(embeds, Path(save_path, 'embeds.pt'))
                    np.save(str(Path(save_path, 'classes.npy')), classes)
                return True
                
            elif isinstance(source, list):
                #option 2: list of audio in numpy format
                mean_embed = None
                embed = None
                for audio_data_np in source:
                    embed = self.embed_utterance(audio_data_np, 
                                                 eval_frames=eval_frames, 
                                                 num_eval=num_eval,
                                                 normalize=self.__model__.test_normalize,
                                                 sr=8000)
                    if mean_embed is None:
                        mean_embed = embed.unsqueeze(0)
                    else:
                        mean_embed = torch.cat(
                            (mean_embed, embed.unsqueeze(0)), 0)
                mean_embed = torch.mean(mean_embed, dim=0)
                
                if save_path:
                    torch.save(mean_embed, Path(save_path, 'embeds.pt'))
                return mean_embed                
        else:
            raise NotImplementedError
        
    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## get embedding of a single utterance
    ## ===== ===== ===== ===== ===== ===== ===== =====
    def embed_utterance(self,
                        source,
                        eval_frames=0,
                        num_eval=10,
                        normalize=False, sr=None):
        """
        Get embedding from utterance
        """
        audio = loadWAV(source,
                        eval_frames,
                        evalmode=True,
                        num_eval=num_eval,
                        sr=sr)

        inp = torch.FloatTensor(audio).to(self.device)
                    
        with torch.no_grad():
            embed = self.__model__.forward(inp).detach().cpu()
        if normalize:
            embed = F.normalize(embed, p=2, dim=1)
        return embed

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Save parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def saveParameters(self, path):

        torch.save(self.__model__.module.state_dict(), path)

    def loadParameters(self, path, show_error=True):
        self_state = self.__model__.module.state_dict()
        
        if self.device == torch.device('cpu'):
            loaded_state = torch.load(path, map_location=torch.device('cpu'))
        else:
            print(f"Load model in {torch.cuda.device_count()} GPU(s)")
            loaded_state = torch.load(path, map_location="cuda:%d" % self.gpu)
            
        for name, param in loaded_state.items():
            origname = name
            
            if name not in self_state:
                name = name.replace("module.", "")

                if name not in self_state:
                    if show_error:
                        print("{} is not in the model.".format(origname))
                    continue

            if self_state[name].size() != loaded_state[origname].size():
                if show_error:
                    print("Wrong parameter length: {}, model: {}, loaded: {}".format(
                        origname, self_state[name].size(), loaded_state[origname].size()))
                continue

            self_state[name].copy_(param)

    def export_onnx(self, state_path, check=True):
        save_root = self.save_path + f"/{self.model_name}/model"
        save_path = os.path.join(save_root, f"model_eval_{self.model_name}.onnx")

        input_names = ["input"]
        output_names = ["output"]
        # NOTE: because using torch.audio -> cant export onnx
        # nume_val, samplerate * (1 + win_len - hoplen)
        dummy_input = torch.randn(10, 8120, device="cuda")

        self.loadParameters(state_path)
        self.__model__.eval()

        torch.onnx.export(self.__S__,
                          dummy_input,
                          save_path,
                          verbose=False,
                          input_names=input_names,
                          output_names=output_names,
                          
                          export_params=True,
                          opset_version=11)

        # double check
        if os.path.exists(save_path) and check:
            print("checking export")
            model = onnx.load(save_path)
            onnx.checker.check_model(model)
            print(onnx.helper.printable_graph(model.graph))
            cprint("Done!!!", 'r')

    def onnx_inference(self, model_path, inp):
        def to_numpy(tensor):
            if not torch.is_tensor(tensor):
                tensor = torch.FloatTensor(tensor)
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
        
        onnx_session = onnxrt.InferenceSession(model_path)
        onnx_inputs = {onnx_session.get_inputs()[0].name: to_numpy(inp)}
        onnx_output = onnx_session.run(None, onnx_inputs)
        return onnx_output
##################################################################################################