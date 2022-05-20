import os
import argparse
import time

import numpy as np
from sklearn.utils import shuffle
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from processing.audio_loader import loadWAV, AugmentWAV
from utils import round_down, worker_init_fn
import torch.distributed as dist
from utils import read_config


class TrainLoader(Dataset):
    def __init__(self, dataset_file_name,
                 augment,
                 augment_options,
                 audio_spec,
                 aug_folder='offline'):

        self.dataset_file_name = dataset_file_name
        self.audio_spec = audio_spec
        self.augment_options = augment_options
        self.max_frames = round(audio_spec['sample_rate'] * (
            audio_spec['sentence_len'] - audio_spec['win_len']) / audio_spec['hop_len'])
        self.augment = augment

        self.sr = audio_spec['sample_rate']

        # augmented folder files
        self.aug_folder = aug_folder
        self.augment_paths = augment_options['augment_paths']
        self.augment_chain = augment_options['augment_chain']

        if self.augment and ('env_corrupt' in self.augment_chain):
            if all(os.path.exists(path) for path in [self.musan_path, self.rir_path]):
                self.augment_engine = AugmentWAV(
                    augment_options, audio_spec, target_db=None)
            else:
                self.augment_engine = None

        # Read Training Files...
        with open(dataset_file_name) as dataset_file:
            lines = dataset_file.readlines()

        dictkeys = list(set([x.split()[0] for x in lines]))
        dictkeys.sort()
        dictkeys = {key: ii for ii, key in enumerate(dictkeys)}

        self.label_dict = {}
        self.data_list = []
        self.data_label = []

        for lidx, line in enumerate(lines):
            data = line.strip().split()

            speaker_label = dictkeys[data[0]]

            if not (speaker_label in self.label_dict):
                self.label_dict[speaker_label] = []

            self.label_dict[speaker_label].append(lidx)
            self.data_label.append(speaker_label)
            self.data_list.append(data[1])

    def __getitem__(self, indices):
        feat = []

        for index in indices:
            # Load audio
            audio_file = self.data_list[index]
            # time domain augment
            audio = loadWAV(audio_file, self.audio_spec,
                            evalmode=False,
                            augment=self.augment,
                            augment_options=self.augment_options)

            # env corrupt augment
            if self.augment and ('env_corrupt' in self.augment_chain) and (self.aug_folder == 'online'):
                # if exists augmented folder(30GB) separately
                # env corruption adding from musan, revberation
                env_corrupt_proportions = self.augment_options['noise_proportion']
                augtype = np.random.choice(
                    ['rev', 'noise', 'both', 'none'], p=[0.2, 0.4, 0.2, 0.2])
                if augtype == 'rev':
                    audio = self.augment_engine.reverberate(audio)
                elif augtype == 'noise':
                    mode = np.random.choice(
                        ['noise', 'speech', 'music', 'noise_vad', 'noise_rirs'], p=env_corrupt_proportions)
                    audio = self.augment_engine.additive_noise(mode, audio)
                elif augtype == 'both':
                    # combined reverb and noise
                    order = np.random.choice(
                        ['noise_first', 'rev_first'], p=[0.5, 0.5])
                    if order == 'rev_first':
                        audio = self.augment_engine.reverberate(audio)
                        mode = np.random.choice(
                            ['noise', 'speech', 'music', 'noise_vad', 'noise_rirs'], p=env_corrupt_proportions)
                        audio = self.augment_engine.additive_noise(mode, audio)
                    else:
                        mode = np.random.choice(
                            ['noise', 'speech', 'music', 'noise_vad', 'noise_rirs'], p=env_corrupt_proportions)
                        audio = self.augment_engine.additive_noise(mode, audio)
                        audio = self.augment_engine.reverberate(audio)
                else:
                    # none type means dont augment
                    pass

            feat.append(audio)

        feat = np.concatenate(feat, axis=0)

        return torch.FloatTensor(feat), self.data_label[index]

    def __len__(self):
        return len(self.data_list)


class TrainSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, nPerSpeaker, max_seg_per_spk, batch_size, **kwargs):
        self.data_source = data_source
        self.label_dict = data_source.label_dict
        self.nPerSpeaker = nPerSpeaker
        self.max_seg_per_spk = max_seg_per_spk
        self.batch_size = batch_size

    def __iter__(self):
        dictkeys = list(self.label_dict.keys())
        dictkeys.sort()

        def lol(lst, sz): return [lst[i:i + sz]
                                  for i in range(0, len(lst), sz)]

        flattened_list = []
        flattened_label = []

        # Data for each class
        for findex, key in enumerate(dictkeys):
            data = self.label_dict[key]
            numSeg = round_down(min(len(data), self.max_seg_per_spk),
                                self.nPerSpeaker)

            rp = lol(
                np.random.permutation(len(data))[:numSeg], self.nPerSpeaker)
            flattened_label.extend([findex] * (len(rp)))
            for indices in rp:
                flattened_list.append([data[i] for i in indices])

        # Data in random order
        mixid = np.random.permutation(len(flattened_label))
        mixlabel = []
        mixmap = []

        # Prevent two pairs of the same speaker in the same batch
        for ii in mixid:
            startbatch = len(mixlabel) - len(mixlabel) % self.batch_size
            if flattened_label[ii] not in mixlabel[startbatch:]:
                mixlabel.append(flattened_label[ii])
                mixmap.append(ii)

        return iter([flattened_list[i] for i in mixmap])

    def __len__(self):
        return len(self.data_source)


def train_data_loader(args):
    train_annotation = args.train_annotation

    batch_size = args.dataloader_options['batch_size']
    shuffle = args.dataloader_options['shuffle']
    num_workers = args.dataloader_options['num_workers']
    nPerSpeaker = args.dataloader_options['nPerSpeaker']
    max_seg_per_spk = args.dataloader_options['max_seg_per_spk']

    augment = args.augment

    train_dataset = TrainLoader(train_annotation,
                                augment,
                                args.augment_options,
                                args.audio_spec,
                                aug_folder='online')

    train_sampler = TrainSampler(train_dataset, nPerSpeaker,
                                 max_seg_per_spk, batch_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=train_sampler,
        pin_memory=False,
        worker_init_fn=worker_init_fn,
        drop_last=True,
        shuffle=shuffle,
    )

    return train_loader


def test_data_loader():
    pass


class test_dataset_loader(Dataset):
    def __init__(self, test_list, test_path, eval_frames, num_eval, **kwargs):
        self.max_frames = eval_frames
        self.num_eval = num_eval
        self.test_path = test_path
        self.test_list = test_list

    def __getitem__(self, index):
        audio = loadWAV(os.path.join(
            self.test_path, self.test_list[index]), self.max_frames, evalmode=True, num_eval=self.num_eval)
        return torch.FloatTensor(audio), self.test_list[index]

    def __len__(self):
        return len(self.test_list)


# Test Data Loader
parser = argparse.ArgumentParser(description="Data loader")
if __name__ == '__main__':
    # Test for data loader
    # YAML
    parser.add_argument('--config', type=str, default=None)

    # control flow
    parser.add_argument('--do_train', action='store_true', default=False)
    parser.add_argument('--do_infer', action='store_true', default=False)
    parser.add_argument('--do_export', action='store_true', default=False)

    # Infer mode
    parser.add_argument('--eval',
                        dest='eval',
                        action='store_true',
                        help='Eval only')
    parser.add_argument('--test',
                        dest='test',
                        action='store_true',
                        help='Test only')
    parser.add_argument('--predict',
                        dest='predict',
                        action='store_true',
                        help='Predict')

    # Device settings
    parser.add_argument('--device',
                        type=str,
                        default="cuda",
                        help='cuda or cpu')
    parser.add_argument('--distributed',
                        action='store_true',
                        default=True,
                        help='Decise wether use multi gpus')

    # Distributed and mixed precision training
    parser.add_argument('--port',
                        type=str,
                        default="8888",
                        help='Port for distributed training, input as text')
    parser.add_argument('--mixedprec',
                        dest='mixedprec',
                        action='store_true',
                        help='Enable mixed precision training')

    parser.add_argument('--nDataLoaderThread',
                        type=int,
                        default=2,
                        help='# of loader threads')

    parser.add_argument('--augment',
                        action='store_true',
                        default=False,
                        help='Augment input')

    parser.add_argument('--early_stopping',
                        action='store_true',
                        default=False,
                        help='Early stopping')

    parser.add_argument('--seed',
                        type=int,
                        default=1000,
                        help='seed')
  #--------------------------------------------------------------------------------------#

    sys_args = parser.parse_args()

    if sys_args.config is not None:
        args = read_config(sys_args.config, sys_args)
        args = argparse.Namespace(**args)

    t = time.time()
    train_loader = train_data_loader(args)

    print("Delay: ", time.time() - t)
    print(len(train_loader))

    for (sample, label) in tqdm(train_loader):
        sample = sample.transpose(0, 1)
        for inp in sample:
            print(inp.size())
        print(sample.size(), label.size())
